document.addEventListener("DOMContentLoaded", function () {
    const i1 = document.getElementById('i1');
    const i2 = document.getElementById('i2');
    const i3 = document.getElementById('i3');
    const i4 = document.getElementById('i4');
    const i5 = document.getElementById('i5');
  
    // Play the first video
    i1.play();
  
    // Set a timer to switch videos after 10 seconds
    setTimeout(() => {
      // Pause the first video
      i1.pause();
      // Hide the first video
      i1.classList.add('hidden');
  
      // Show and play the second video
      i2.classList.remove('hidden');
      i2.play();
    }, 10000); // 10000 milliseconds = 10 seconds
  });
  