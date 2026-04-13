<script>
  export let username = '';
  export let password = '';
  export let status = '';
  export let onLogin = () => {};
</script>

<h1>Login</h1>
<input placeholder="Username" bind:value={username} />
<input placeholder="Universal password" type="password" bind:value={password} />
<button on:click={onLogin}>Login</button>

{#if status}
  <p class="status">{status}</p>
{/if}

<style>
  h1 {
    margin-top: 0;
  }

  input {
    display: block;
    width: 100%;
    padding: 0.5rem;
    margin-bottom: 0.6rem;
    border: 1px solid #aaa;
    box-sizing: border-box;
  }

  button {
    padding: 0.5rem 0.8rem;
    margin-right: 0.5rem;
    cursor: pointer;
    border: 1px solid #aaa;
    background: #fafafa;
  }

  .status {
    margin-top: 0.8rem;
  }
</style>
