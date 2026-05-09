export function createStatusMessages(runMessage) {
  let runMessageTimeout = null;

  function setRunMessage(value, temporary = false) {
    runMessage.set(value);
    if (runMessageTimeout) {
      clearTimeout(runMessageTimeout);
      runMessageTimeout = null;
    }

    if (temporary && value) {
      runMessageTimeout = setTimeout(() => {
        runMessage.set('');
        runMessageTimeout = null;
      }, 3000);
    }
  }

  function setStatusMessage(value) {
    setRunMessage(value, false);
  }

  function setTempStatusMessage(value) {
    setRunMessage(value, true);
  }

  return {
    setRunMessage,
    setStatusMessage,
    setTempStatusMessage
  };
}
