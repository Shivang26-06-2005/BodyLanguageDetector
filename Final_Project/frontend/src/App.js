import React from "react";
import Navbar from "./components/Navbar";
import Home from "./components/Home";
import StartInterview from "./components/StartInterview";

export default function App() {
  return (
    <div>
      <Navbar />
      <Home />
      <StartInterview />
    </div>
  );
}
