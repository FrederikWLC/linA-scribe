<script>
  import { onMount } from 'svelte';
  import LoginPage from './pages/LoginPage.svelte';
  import ToolPage from './pages/ToolPage.svelte';

  let username = '';
  let password = '';
  let currentUser = '';
  let token = '';
  let route = '/login';
  let status = '';

  function goTo(path, replace = false) {
    route = path;
    if (replace) {
      window.history.replaceState({}, '', path);
    } else {
      window.history.pushState({}, '', path);
    }
  }

  function syncRoute(pathname, replace = true) {
    const normalized = pathname === '/tool' ? '/tool' : '/login';

    if (normalized === '/tool' && !currentUser) {
      status = 'Please login first';
      goTo('/login', true);
      return;
    }

    if (normalized === '/login' && currentUser) {
      goTo('/tool', true);
      return;
    }

    goTo(normalized, replace);
  }

  async function login() {
    status = '';
    const cleanUsername = username.trim();
    if (!cleanUsername || !password) {
      status = 'Enter username and password';
      return;
    }

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: cleanUsername, password })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
      }

      currentUser = data.username;
      token = data.token;
      localStorage.setItem('username', currentUser);
      localStorage.setItem('token', token);
      password = '';
      status = `Logged in as ${currentUser}`;
      goTo('/tool', true);
    } catch (err) {
      status = err instanceof Error ? err.message : String(err);
      currentUser = '';
      token = '';
    }
  }

  function logout() {
    currentUser = '';
    token = '';
    localStorage.removeItem('username');
    localStorage.removeItem('token');
    goTo('/login', true);
    status = 'Logged out';
  }

  onMount(() => {
    const saved = localStorage.getItem('username');
    const savedToken = localStorage.getItem('token');
    if (saved && savedToken) {
      currentUser = saved;
      token = savedToken;
    }

    syncRoute(window.location.pathname, true);

    const onPopState = () => syncRoute(window.location.pathname, true);
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  });
</script>

<main>
  {#if route === '/login'}
    <LoginPage bind:username bind:password {status} onLogin={login} />
  {:else}
    <ToolPage {currentUser} {token} onLogout={logout} />
  {/if}
</main>

<style>
  :global(body) {
    margin: 0;
    font-family: sans-serif;
    background: #f7f7f7;
    color: #222;
  }

  main {
    max-width: 760px;
    margin: 2rem auto;
    padding: 1rem;
    background: #fff;
    border: 1px solid #ddd;
  }
</style>
