document.addEventListener('DOMContentLoaded', () => {
  const contentContainer = document.getElementById('plotly-content-container');
  const url = contentContainer.getAttribute('data-url');

  fetch(url)
    .then(response => response.text())
    .then(data => {
      contentContainer.innerHTML = data;
    })
    .catch(error => {
      contentContainer.innerHTML = '<p>Error loading content.</p>';
      console.error('Error loading external content:', error);
    });
});


