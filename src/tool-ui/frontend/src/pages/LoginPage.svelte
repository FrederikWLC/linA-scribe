<script>
  import '../static/css/LoginPage.css';

  export let username = '';
  export let password = '';
  export let status = '';
  export let onLogin = () => {};
</script>

<div class="login-page">
  <h1>Login</h1>
  <input placeholder="Username" bind:value={username} />
  <input placeholder="Universal password" type="password" bind:value={password} />
  <button on:click={onLogin}>Login</button>

  {#if status}
    <p class="status">{status}</p>
  {/if}
</div>
