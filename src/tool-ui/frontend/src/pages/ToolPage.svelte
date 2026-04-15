<script>
  export let currentUser = '';
  export let protectedMessage = '';
  export let onReloadProtected = () => {};
  export let onLogout = () => {};
</script>

<h1>Tool</h1>
<p>Logged in as <strong>{currentUser}</strong></p>
<p>
  <button on:click={onReloadProtected}>Reload Protected Data</button>
  <button on:click={onLogout}>Logout</button>
</p>

<h2>Protected Page</h2>
<p>{protectedMessage}</p>

<style>
  h1 {
    margin-top: 0;
  }

  button {
    padding: 0.5rem 0.8rem;
    margin-right: 0.5rem;
    cursor: pointer;
    border: 1px solid #aaa;
    background: #fafafa;
  }
</style>
