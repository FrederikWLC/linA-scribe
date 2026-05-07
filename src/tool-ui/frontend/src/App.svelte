<script>
  import { onMount } from 'svelte';
  import LoginPage from './pages/LoginPage.svelte';
  import ToolPage from './pages/ToolPage.svelte';

  const ROUTE_FOR_MODEL_ID = {
    sam: '/tool/sam',
    gaussian: '/tool/gaussian'
  };

  let username = '';
  let password = '';
  let currentUser = '';
  let token = '';
  let route = '/login';
  let selectedRouteModelID = '';
  let requestedToolRoute = '';
  let status = '';

  function routeFromPathname(pathname) {
    const clean = pathname.replace(/\/+$/, '').replace(/\/+/g, '/');
    if (clean === '/tool') {
      return '/tool';
    }

    const match = clean.match(/^\/tool\/(sam|gaussian)$/);
    if (match) {
      return `/tool/${match[1]}`;
    }

    return '/tool';
  }

  function modelIDFromRoute(routePath) {
    const match = routePath.match(/^\/tool\/(sam|gaussian)$/);
    return match ? match[1] : '';
  }

  function goTo(path, replace = false) {
    route = path;
    selectedRouteModelID = modelIDFromRoute(path);
    if (replace) {
      window.history.replaceState({}, '', path);
    } else {
      window.history.pushState({}, '', path);
    }
  }

  function syncRoute(pathname, replace = true) {
    const normalized = routeFromPathname(pathname);

    if (normalized.startsWith('/tool') && !currentUser) {
      requestedToolRoute = normalized;
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
      goTo(requestedToolRoute || '/tool', true);
      requestedToolRoute = '';
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
    requestedToolRoute = '';
    goTo('/login', true);
    status = 'Logged out';
  }

  function pickTool(routePath) {
    if (routePath !== route) {
      goTo(routePath);
    }
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
  {:else if route === '/tool'}
    <section class="tool-selection">
      <h1>Choose a tool</h1>
      <p>Select one of the two available models.</p>
      <div class="tool-selection-buttons">
        <button type="button" on:click={() => pickTool('/tool/sam')}>SAM</button>
        <button type="button" on:click={() => pickTool('/tool/gaussian')}>Gaussian</button>
      </div>
    </section>
  {:else}
    <ToolPage
      {token}
      routeModelID={selectedRouteModelID}
      onRouteChange={goTo}
    />
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
    margin: 2rem 0;
  }

  .tool-selection {
    max-width: 760px;
    margin: 0 auto;
    padding: 2rem;
    background: #fff;
    border: 1px solid #ddd;
    text-align: center;
  }

  .tool-selection-buttons {
    display: grid;
    grid-template-columns: repeat(2, minmax(120px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
  }

  .tool-selection-buttons button {
    padding: 0.85rem 1rem;
    font-size: 1rem;
    border: 1px solid #444;
    border-radius: 6px;
    background: #fff;
    cursor: pointer;
  }

  .tool-selection-buttons button:hover {
    background: #f2f2f2;
  }
</style>
