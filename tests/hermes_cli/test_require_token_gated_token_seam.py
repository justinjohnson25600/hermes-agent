"""Behaviour-contract tests: the gated branch of ``web_server._require_token`` must honour
the token-auth seam.

The token-auth seam (``hermes_cli.dashboard_auth.token_auth``) runs outermost and, for a
route registered via ``register_token_route``, verifies the bearer against the registered
token providers, attaches ``request.state.token_principal`` + ``token_authenticated``, and
401s/503s everything else itself. Every auth layer above ``_require_token`` honours that
flag — ``gated_auth_middleware`` (dashboard_auth/middleware.py) and ``auth_middleware``
(web_server.py) both let a ``token_authenticated`` request through — but the gated branch
of ``_require_token`` only accepted ``request.state.session`` (a cookie/OAuth session), so
a service caller on a token route reached the handler and was still 401'd by it.

That is the same-OS-user maintenance path the product publishes its rendezvous token for
(``host-serve.token``, 0600; see the ``publish_record(..., token=_SESSION_TOKEN)`` call in
``hermes_cli/web_server.py``): ``gateway/host_rendezvous.probe_owner`` dials
``GET /api/host/identity`` with ``Authorization: Bearer <token>`` and attaches only when
the owner answers — on a gated backend that probe always 401'd.

These tests exercise the real middleware stack (host-header -> OAuth gate -> legacy auth
middleware -> token seam, as installed on ``web_server.app``) and assert the contract:

  * a valid bearer on a registered token route reaches the handler (not 401) in gated
    mode — and the response carries the handler's own payload, not an error body;
  * a wrong bearer still 401s (the seam fails closed; ``_require_token`` must not widen
    that);
  * the legacy ``_SESSION_TOKEN`` still does NOT unlock gated machine routes (a request
    that never went through the seam stays 401);
  * in loopback mode the legacy session-token path is untouched (200 with the header);
  * a cookie-authenticated browser session still reaches ``_require_token`` handlers
    (the original gated regression, #install-popup class).
"""
from __future__ import annotations

import hmac
import secrets

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    LoginStart,
    Session,
    TokenPrincipal,
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth import token_auth
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

# The machine-only ``_require_token`` routes a maintenance client needs. Production
# token providers opt routes in via ``register_token_route`` (exactly as the drain
# plugin does for ``/api/gateway/drain``), and the seam only authenticates registered
# paths — so the fixture registers all three, the way the host-owner product does.
MACHINE_TOKEN_ROUTES = (
    "/api/host/identity",
    "/api/health/idle",
    "/api/health/retirement",
)
IDENTITY_ROUTE = MACHINE_TOKEN_ROUTES[0]

_TEST_SECRET = "test-rendezvous-secret-0123456789abcdef"


class _BearerOnlyProvider(DashboardAuthProvider):
    """Non-interactive verifier for the host owner's rendezvous credential.

    Token capability only (like the drain provider): verifies exactly one secret,
    constant-time, and never issues sessions.
    """

    name = "host-owner-test"
    display_name = "Host Owner (test)"

    supports_token = True
    supports_session = False

    def __init__(self, *, secret: str = _TEST_SECRET):
        self._secret = secret

    def verify_token(self, *, token: str):
        if token and hmac.compare_digest(
            token.encode("utf-8"), self._secret.encode("utf-8")
        ):
            return TokenPrincipal(
                principal="host-owner", provider=self.name, scopes=("host-owner",)
            )
        return None

    # -- interactive capability: unsupported, fail loud -----------------------

    def start_login(self, *, redirect_uri: str):
        raise NotImplementedError("machine credential: no login flow")

    def complete_login(self, *, code, state, code_verifier, redirect_uri):
        raise NotImplementedError("machine credential: no login flow")

    def verify_session(self, *, access_token: str):
        return None

    def refresh_session(self, *, refresh_token: str):
        raise NotImplementedError("machine credential: no sessions")

    def revoke_session(self, *, refresh_token: str):
        return None


@pytest.fixture
def gated_app_with_token_route():
    """Gated app + the identity route opted into the token-auth seam.

    Mirrors the ``gated_app`` fixture in ``test_dashboard_auth_middleware.py`` and
    registers a stub OAuth provider so the cookie path stays exercisable; additionally
    registers a bearer-only provider and the token route under test.
    """
    clear_providers()
    register_provider(StubAuthProvider())
    register_provider(_BearerOnlyProvider())
    for route in MACHINE_TOKEN_ROUTES:
        token_auth.register_token_route(route)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    token_auth.clear_token_routes()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _complete_stub_login(client) -> None:
    """Walk the stub OAuth round trip so the client carries a valid cookie session."""
    r1 = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert r1.status_code == 302
    state = r1.headers["location"].split("state=")[1]
    r2 = client.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    assert r2.status_code == 302


def test_gated_token_route_bearer_reaches_require_token_handler(
    gated_app_with_token_route,
):
    """A seam-authenticated bearer on a registered token route must clear the gated
    branch of ``_require_token`` and get the handler's payload — this is the machine
    path ``probe_owner`` and any same-OS-user maintenance client depends on."""
    r = gated_app_with_token_route.get(
        IDENTITY_ROUTE, headers={"Authorization": f"Bearer {_TEST_SECRET}"}
    )
    assert r.status_code == 200, (
        "A token-auth-seam-authenticated request was rejected by the gated branch of "
        f"_require_token (status {r.status_code}): {r.text}"
    )
    body = r.json()
    # Reached the real handler: it answers with the owner identity payload, not an
    # auth error body.
    assert body.get("ok") is True
    assert "pid" in body and "role" in body


@pytest.mark.parametrize("route", ["/api/health/idle", "/api/health/retirement"])
def test_gated_health_routes_accept_seam_bearer(gated_app_with_token_route, route):
    """The other machine-only ``_require_token`` routes accept the same bearer.

    ``/api/health/retirement`` is probed with a deliberately invalid action: a
    passing auth layer lets the request reach the handler, which rejects the
    action with its own 400 (never invoke a real retirement action from a
    test). The contract is "not 401"; the handler status is route-specific.
    """
    if route == "/api/health/idle":
        r = gated_app_with_token_route.get(
            route, headers={"Authorization": f"Bearer {_TEST_SECRET}"}
        )
        assert r.status_code == 200, (
            f"{route} rejected a seam-authenticated maintenance request "
            f"(status {r.status_code}): {r.text}"
        )
    else:
        r = gated_app_with_token_route.post(
            route,
            json={"action": "bogus-action"},
            headers={"Authorization": f"Bearer {_TEST_SECRET}"},
        )
        assert r.status_code != 401, (
            f"{route} 401'd a seam-authenticated maintenance request: {r.text}"
        )
        # Reached the handler's own action validation.
        assert r.status_code == 400, (
            f"Expected the retirement handler's 400 (bad action), got "
            f"{r.status_code}: {r.text}"
        )


def test_gated_token_route_wrong_bearer_still_401(gated_app_with_token_route):
    """The seam fails closed; ``_require_token`` must never widen that."""
    r = gated_app_with_token_route.get(
        IDENTITY_ROUTE, headers={"Authorization": f"Bearer {secrets.token_urlsafe(32)}"}
    )
    assert r.status_code == 401, (
        f"A wrong bearer on a gated token route must stay 401, got {r.status_code}"
    )


def test_gated_token_route_no_bearer_still_401(gated_app_with_token_route):
    """No token and no cookie in gated mode: still bounced (401, not a /login redirect
    for an /api path)."""
    r = gated_app_with_token_route.get(IDENTITY_ROUTE, follow_redirects=False)
    assert r.status_code == 401, (
        f"An unauthenticated request on a gated token route must stay 401, "
        f"got {r.status_code}"
    )


def test_gated_legacy_session_token_does_not_unlock_machine_route(
    gated_app_with_token_route,
):
    """The legacy ``_SESSION_TOKEN`` must NOT unlock a gated machine route: on a
    seam-registered route the seam 401s a bearer-less request before any handler
    runs, so the static header alone stays insufficient (the fix must not turn the
    legacy token into a machine credential)."""
    r = gated_app_with_token_route.get(
        IDENTITY_ROUTE,
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )
    assert r.status_code == 401, (
        f"The legacy session-token header must not unlock a gated machine route, "
        f"got {r.status_code}"
    )


def test_loopback_seam_bearer_accepted_on_registered_machine_route(
    gated_app_with_token_route,
):
    """Loopback mode honours the seam too (``auth_middleware`` already skips
    ``token_authenticated`` requests): a verified bearer on a registered route
    clears ``_require_token``'s loopback branch instead of failing the legacy
    token check."""
    web_server.app.state.auth_required = False
    try:
        r = gated_app_with_token_route.get(
            IDENTITY_ROUTE, headers={"Authorization": f"Bearer {_TEST_SECRET}"}
        )
        assert r.status_code == 200, (
            f"A seam-authenticated bearer was rejected on a registered machine route "
            f"in loopback mode: {r.status_code} {r.text}"
        )
        assert r.json().get("ok") is True
    finally:
        web_server.app.state.auth_required = True


def test_loopback_wrong_bearer_still_rejected(gated_app_with_token_route):
    """Loopback mode keeps failing closed: a wrong bearer stays 401."""
    web_server.app.state.auth_required = False
    try:
        r = gated_app_with_token_route.get(
            IDENTITY_ROUTE, headers={"Authorization": "Bearer not-the-token"}
        )
        assert r.status_code == 401, (
            f"A wrong bearer must stay 401 in loopback mode, got {r.status_code}"
        )
    finally:
        web_server.app.state.auth_required = True


def test_loopback_session_token_still_accepted_on_unregistered_require_token_route(
    gated_app_with_token_route,
):
    """The legacy loopback contract is untouched on routes NOT registered with the
    seam: the SPA path (session-token header) still authorizes."""
    web_server.app.state.auth_required = False
    try:
        r = gated_app_with_token_route.get(
            "/api/dashboard/plugins/hub",
            headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
        )
        assert r.status_code == 200, (
            f"Loopback session-token path regressed: {r.status_code} {r.text}"
        )
    finally:
        web_server.app.state.auth_required = True


def test_gated_cookie_session_still_reaches_require_token_handler(
    gated_app_with_token_route,
):
    """The original gated regression stays fixed: a cookie-authenticated browser
    session still clears ``_require_token``. Uses a NON-token-registered route (the
    SPA's plugin hub): a token route is owned by the seam, which 401s a cookie-only
    browser request by design — that route-ownership contract is asserted separately
    below."""
    _complete_stub_login(gated_app_with_token_route)
    r = gated_app_with_token_route.get("/api/dashboard/plugins/hub")
    assert r.status_code != 401, (
        f"A cookie-authenticated request was 401'd by _require_token under the gate "
        f"(the install-popup bug class): {r.status_code} {r.text}"
    )


def test_cookie_only_browser_request_to_token_route_is_rejected(
    gated_app_with_token_route,
):
    """Route ownership is exclusive: once a path is registered with the seam, a
    cookie-only browser request is 401'd by the seam itself (it never reaches
    ``_require_token``). The fix must not disturb this contract."""
    _complete_stub_login(gated_app_with_token_route)
    r = gated_app_with_token_route.get(IDENTITY_ROUTE)
    assert r.status_code == 401, (
        f"A cookie-only request to a token-registered route must be rejected by the "
        f"seam (route-ownership contract), got {r.status_code}: {r.text}"
    )


def test_identity_payload_shape_contract(gated_app_with_token_route):
    """The identity payload ``probe_owner`` matches on: ``pid`` (owner's live pid),
    ``role`` and the ``servesSpa`` flag — plus ``protocolVersion`` for the handshake."""
    r = gated_app_with_token_route.get(
        IDENTITY_ROUTE, headers={"Authorization": f"Bearer {_TEST_SECRET}"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["pid"] > 0
    assert body["protocolVersion"] == 1
    assert isinstance(body["servesSpa"], bool)
