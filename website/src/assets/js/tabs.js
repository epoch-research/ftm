function init_tabs(update_url) {
  for (let span of document.querySelectorAll('.tabs .tab-links span')) {
    span.addEventListener('click', function(e) {
      let currentAttrValue = this.dataset.href;

      let tabs = this.parentElement.parentElement.parentElement;

      // Show/Hide Tabs
      for (let element of tabs.querySelectorAll(`[data-id]`)) {
        element.classList.remove('active');
      }

      tabs.querySelector(`[data-id="${currentAttrValue.slice(1)}"]`).classList.add('active');

      // Change/remove current tab to active
      for (let element of tabs.querySelectorAll(`.tab-links li`)) {
        element.classList.remove('active');
      }

      this.parentElement.classList.add('active');

      if (update_url) {
        history.replaceState(null, null, currentAttrValue);
      }

      e.preventDefault();
    });
  }
}
