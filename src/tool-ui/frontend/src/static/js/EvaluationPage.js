export function getEvaluationProcesses(handlers, states) {
  return [
    {
      title: 'Tuning Files',
      state: states.tuningState,
      onRefresh: handlers.onLoadTuningFiles
    },
    {
      title: 'Evaluation Files',
      state: states.evaluationState,
      onRefresh: handlers.onLoadEvaluationFiles
    },
    {
      title: 'Ablation Files',
      state: states.ablationState,
      onRefresh: handlers.onLoadAblationFiles
    },
    {
      title: 'Scribing Files',
      state: states.scribingState,
      onRefresh: handlers.onLoadScribingFiles
    }
  ];
}
