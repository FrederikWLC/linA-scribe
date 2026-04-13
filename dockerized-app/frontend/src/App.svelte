<script>
  import { onMount } from 'svelte';
  import LoginPage from './pages/LoginPage.svelte';
  import ToolPage from './pages/ToolPage.svelte';
  import EvaluationPage from './pages/EvaluationPage.svelte';

  let username = '';
  let password = '';
  let currentUser = '';
  let token = '';
  let role = '';
  let route = '/login';
  let status = '';
  let protectedMessage = '';
  let evaluationMessage = '';

  let tuningState = { message: '', files: [] };
  let evaluationState = { message: '', files: [] };
  let ablationState = { message: '', files: [] };
  let scribingState = { message: '', files: [] };

  function goTo(path, replace = false) {
    route = path;
    if (replace) {
      window.history.replaceState({}, '', path);
    } else {
      window.history.pushState({}, '', path);
    }
  }

  function syncRoute(pathname, replace = true) {
    const normalized = pathname === '/evaluation' ? '/evaluation' : pathname === '/tool' ? '/tool' : '/login';

    if ((normalized === '/tool' || normalized === '/evaluation') && !currentUser) {
      status = 'Please login first';
      goTo('/login', true);
      return;
    }

    if (normalized === '/login' && currentUser) {
      goTo('/tool', true);
      return;
    }

    if (normalized === '/evaluation' && role !== 'master') {
      status = 'Master access required';
      goTo('/tool', true);
      return;
    }

    goTo(normalized, replace);

    if (normalized === '/tool' && currentUser) {
      loadProtectedPage();
    }

    if (normalized === '/evaluation' && currentUser) {
      loadEvaluationPage();
    }
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
      role = data.role;
      localStorage.setItem('username', currentUser);
      localStorage.setItem('token', token);
      localStorage.setItem('role', role);
      password = '';
      status = data.created ? `User created and logged in as ${role}` : `Logged in as ${role}`;
      goTo(role === 'master' ? '/evaluation' : '/tool', true);
    } catch (err) {
      status = err instanceof Error ? err.message : String(err);
      currentUser = '';
      token = '';
      role = '';
    }
  }

  async function loadProtectedPage() {
    protectedMessage = '';
    try {
      const response = await fetch('/api/protected/page', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Unauthorized');
      }
      protectedMessage = data.message;
    } catch (err) {
      protectedMessage = err instanceof Error ? err.message : String(err);
    }
  }

  async function loadEvaluationPage() {
    evaluationMessage = '';
    try {
      const response = await fetch('/api/evaluation/page', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Unauthorized');
      }
      evaluationMessage = data.message;
      await loadTuningFiles();
      await loadEvaluationFiles();
      await loadAblationFiles();
      await loadScribingFiles();
    } catch (err) {
      evaluationMessage = err instanceof Error ? err.message : String(err);
    }
  }

  async function loadTuningFiles() {
    tuningState = { ...tuningState, message: 'Loading...' };
    try {
      const response = await fetch('/api/evaluation/tuning/files', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to load tuning files');
      }
      const files = Array.isArray(data.files) ? data.files : [];
      tuningState = { files, message: `Found ${files.length} file(s)` };
    } catch (err) {
      tuningState = { ...tuningState, message: err instanceof Error ? err.message : String(err) };
    }
  }

  async function loadEvaluationFiles() {
    evaluationState = { ...evaluationState, message: 'Loading...' };
    try {
      const response = await fetch('/api/evaluation/evaluation/files', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to load evaluation files');
      }
      const files = Array.isArray(data.files) ? data.files : [];
      evaluationState = { files, message: `Found ${files.length} file(s)` };
    } catch (err) {
      evaluationState = { ...evaluationState, message: err instanceof Error ? err.message : String(err) };
    }
  }

  async function loadAblationFiles() {
    ablationState = { ...ablationState, message: 'Loading...' };
    try {
      const response = await fetch('/api/evaluation/ablation/files', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to load ablation files');
      }
      const files = Array.isArray(data.files) ? data.files : [];
      ablationState = { files, message: `Found ${files.length} file(s)` };
    } catch (err) {
      ablationState = { ...ablationState, message: err instanceof Error ? err.message : String(err) };
    }
  }

  async function loadScribingFiles() {
    scribingState = { ...scribingState, message: 'Loading...' };
    try {
      const response = await fetch('/api/evaluation/scribing/files', {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to load scribing files');
      }
      const files = Array.isArray(data.files) ? data.files : [];
      scribingState = { files, message: `Found ${files.length} file(s)` };
    } catch (err) {
      scribingState = { ...scribingState, message: err instanceof Error ? err.message : String(err) };
    }
  }

  function logout() {
    currentUser = '';
    token = '';
    role = '';
    localStorage.removeItem('username');
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    protectedMessage = '';
    evaluationMessage = '';
    tuningState = { message: '', files: [] };
    evaluationState = { message: '', files: [] };
    ablationState = { message: '', files: [] };
    scribingState = { message: '', files: [] };
    goTo('/login', true);
    status = 'Logged out';
  }

  onMount(() => {
    const saved = localStorage.getItem('username');
    const savedToken = localStorage.getItem('token');
    const savedRole = localStorage.getItem('role');
    if (saved && savedToken && savedRole) {
      currentUser = saved;
      token = savedToken;
      role = savedRole;
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
    <ToolPage
      {currentUser}
      {protectedMessage}
      onReloadProtected={loadProtectedPage}
      onLogout={logout}
    />
  {:else}
    <EvaluationPage
      {currentUser}
      {evaluationMessage}
      {tuningState}
      {evaluationState}
      {ablationState}
      {scribingState}
      onReloadEvaluation={loadEvaluationPage}
      onLoadTuningFiles={loadTuningFiles}
      onLoadEvaluationFiles={loadEvaluationFiles}
      onLoadAblationFiles={loadAblationFiles}
      onLoadScribingFiles={loadScribingFiles}
      onLogout={logout}
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
    max-width: 760px;
    margin: 2rem auto;
    padding: 1rem;
    background: #fff;
    border: 1px solid #ddd;
  }
</style>
