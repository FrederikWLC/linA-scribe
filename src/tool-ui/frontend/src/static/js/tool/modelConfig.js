export const modelOptions = [
  {
    key: 'modal-mobilesam',
    label: 'BestMobileSAMv2Implementation',
    requiresSetImage: true,
    acceptsPrompts: true
  },
  { key: 'gaussian', label: 'Gaussian', requiresSetImage: false, acceptsPrompts: false },
  { key: 'grabcut-auto-brush', label: 'GC+brush', requiresSetImage: false, acceptsPrompts: false }
];

export function resolveInitialModelKey(initialModelKey) {
  return modelOptions.some((model) => model.key === initialModelKey)
    ? initialModelKey
    : 'modal-mobilesam';
}
