<script>
  export let currentUser = '';
  export let evaluationMessage = '';
  export let tuningState = { message: '', files: [] };
  export let evaluationState = { message: '', files: [] };
  export let ablationState = { message: '', files: [] };
  export let scribingState = { message: '', files: [] };
  export let onReloadEvaluation = () => {};
  export let onLoadTuningFiles = () => {};
  export let onLoadEvaluationFiles = () => {};
  export let onLoadAblationFiles = () => {};
  export let onLoadScribingFiles = () => {};
  export let onLogout = () => {};
</script>

<h1>Evaluation</h1>
<p>Master user: <strong>{currentUser}</strong></p>
<p>
  <button on:click={onReloadEvaluation}>Reload All</button>
  <button on:click={onLogout}>Logout</button>
</p>

<h2>Master-Only Page</h2>
<p>{evaluationMessage}</p>

<section class="process-card">
  <h3>Tuning Files</h3>
  <p><button on:click={onLoadTuningFiles}>Refresh</button></p>
  {#if tuningState.message}
    <p class="status">{tuningState.message}</p>
  {/if}
  {#if (tuningState.files || []).length > 0}
    <ul>
      {#each tuningState.files as item}
        <li><a href={item.download_url} target="_blank" rel="noreferrer">{item.path}</a></li>
      {/each}
    </ul>
  {/if}
</section>

<section class="process-card">
  <h3>Evaluation Files</h3>
  <p><button on:click={onLoadEvaluationFiles}>Refresh</button></p>
  {#if evaluationState.message}
    <p class="status">{evaluationState.message}</p>
  {/if}
  {#if (evaluationState.files || []).length > 0}
    <ul>
      {#each evaluationState.files as item}
        <li><a href={item.download_url} target="_blank" rel="noreferrer">{item.path}</a></li>
      {/each}
    </ul>
  {/if}
</section>

<section class="process-card">
  <h3>Ablation Files</h3>
  <p><button on:click={onLoadAblationFiles}>Refresh</button></p>
  {#if ablationState.message}
    <p class="status">{ablationState.message}</p>
  {/if}
  {#if (ablationState.files || []).length > 0}
    <ul>
      {#each ablationState.files as item}
        <li><a href={item.download_url} target="_blank" rel="noreferrer">{item.path}</a></li>
      {/each}
    </ul>
  {/if}
</section>

<section class="process-card">
  <h3>Scribing Files</h3>
  <p><button on:click={onLoadScribingFiles}>Refresh</button></p>
  {#if scribingState.message}
    <p class="status">{scribingState.message}</p>
  {/if}
  {#if (scribingState.files || []).length > 0}
    <ul>
      {#each scribingState.files as item}
        <li><a href={item.download_url} target="_blank" rel="noreferrer">{item.path}</a></li>
      {/each}
    </ul>
  {/if}
</section>

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

  .status {
    margin-top: 0.8rem;
  }

  .process-card {
    border: 1px solid #ddd;
    padding: 0.8rem;
    margin-top: 1rem;
  }

  ul {
    padding-left: 1.2rem;
  }
</style>
