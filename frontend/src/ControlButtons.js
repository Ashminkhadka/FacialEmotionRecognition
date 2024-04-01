import React from 'react';

const ControlButtons = ({ onStart, onStop }) => {
  return (
    <div>
      <button onClick={onStart}>Start Webcam</button>
      <button onClick={onStop}>Stop Webcam</button>
    </div>
  );
};

export default ControlButtons;
