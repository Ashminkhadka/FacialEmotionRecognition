import React, { useState, useEffect } from 'react';
import WebcamFeed from './WebcamFeed';
import ControlButtons from './ControlButtons';
import RadarGraph from './RadarGraph';
import './App.css';

const App = () => {
  const [stream, setStream] = useState(null);
  const [showRadarGraph, setShowRadarGraph] = useState(false);
  const [radarGraphData, setRadarGraphData] = useState(null);

  const startWebcam = async () => {
    const userMedia = await navigator.mediaDevices.getUserMedia({ video: true });
    setStream(userMedia);
  };

  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  };

  const fetchDataFromBackend = async () => {
    try {
      const response = await fetch('http://127.0.0.1:3000/emotion-detection', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to fetch data from the backend');
      }
      const data = await response.json();
      setRadarGraphData(data);
      setShowRadarGraph(true);
    } catch (error) {
      console.error('Error fetching data from backend:', error);
    }
  };

  useEffect(() => {
    if (showRadarGraph) {
      
      fetchDataFromBackend();
    }
  }, [showRadarGraph]);

  return (
    <div>
      <h1>Facial Emotion Recognition for interrogation</h1>
      <WebcamFeed stream={stream} />
      <ControlButtons onStart={startWebcam} onStop={stopWebcam} />
      {showRadarGraph && <RadarGraph data={radarGraphData} />}
    </div>
  );
};

export default App;