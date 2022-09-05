<script>
  // Stolen from mickey li: https://codepen.io/tmac/pen/QgVRKb
</script>

<div class="progress-bar">
  <div class="progress-bar-value" />
</div>

<style>
  .progress-bar {
    height: 8px;
    background-color: rgba(5, 114, 206, 0.2);
    width: 95%;
    overflow: hidden;
    margin: 0 auto;
  }

  .progress-bar-value {
    width: 100%;
    height: 100%;
    background-color: rgb(5, 114, 206);
    animation: indeterminateAnimation 1s infinite linear;
    transform-origin: 0% 50%;
  }

  @keyframes indeterminateAnimation {
    0% {
      transform: translateX(0) scaleX(0);
    }
    40% {
      transform: translateX(0) scaleX(0.4);
    }
    100% {
      transform: translateX(100%) scaleX(0.5);
    }
  }
</style>
