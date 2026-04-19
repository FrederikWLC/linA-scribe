<script>
  import '../static/css/EvaluationPage.css';
  import { getEvaluationProcesses } from '../static/js/EvaluationPage.js';

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

  $: processes = getEvaluationProcesses(
    {
      onLoadTuningFiles,
      onLoadEvaluationFiles,
      onLoadAblationFiles,
      onLoadScribingFiles
    },
    {
      tuningState,
      evaluationState,
      ablationState,
      scribingState
    }
  );
</script>

<div class="evaluation-page">
  <h1>Evaluation</h1>
  <p>Master user: <strong>{currentUser}</strong></p>
  <p>
    <button on:click={onReloadEvaluation}>Reload All</button>
    <button on:click={onLogout}>Logout</button>
  </p>

  <h2>Master-Only Page</h2>
  <p>{evaluationMessage}</p>

  {#each processes as process}
    <section class="process-card">
      <h3>{process.title}</h3>
      <p><button on:click={process.onRefresh}>Refresh</button></p>
      {#if process.state.message}
        <p class="status">{process.state.message}</p>
      {/if}
      {#if (process.state.files || []).length > 0}
        <ul>
          {#each process.state.files as item}
            <li><a href={item.download_url} target="_blank" rel="noreferrer">{item.path}</a></li>
          {/each}
        </ul>
      {/if}
    </section>
  {/each}
</div>
