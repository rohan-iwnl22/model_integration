import { useEffect, useState } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";

// Define a mapping of class indices to category names
const classNames = [
  "BABY PRODUCTS", // index 0
  "BEAUTY HEALTH", // index 1
  "CLOTHING ACCESSORIES JEWELLERY", // index 2
  "ELECTORNICS",
  "GROCERY",
  "HOBBY ARTS AND STATIONARY",
  "HOME KITCHEN TOOLS",
  "PET SUPPLIES",
  "SPORTS OUTDOOR",
];

function App() {
  const [model, setModel] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const loadModelWithInputLayer = async () => {
    try {
      console.log("Starting to load model...");
      let loadedModel = await tf.loadLayersModel("Models/json/model.json");
      console.log("Model loaded successfully:", loadedModel);
      if (!loadedModel.inputs || loadedModel.inputs.length === 0) {
        const inputLayer = tf.input({ shape: [224, 224, 3] });
        const newModel = tf.model({
          inputs: inputLayer,
          outputs: loadedModel.outputs,
        });
        console.log("Model redefined with input layer:", newModel);
        setModel(newModel);
      } else {
        setModel(loadedModel);
      }
    } catch (error) {
      console.error("Detailed error when loading the model:", error);
      setError(`Failed to load the model. Error: ${error.message}`);
    }
  };

  useEffect(() => {
    loadModelWithInputLayer();
  }, []);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const imageURL = URL.createObjectURL(file);
      setImageUrl(imageURL);
    }
  };

  const preprocessImage = (imageElement) => {
    return tf.tidy(() => {
      const tensor = tf.browser
        .fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims();
      return tensor;
    });
  };

  const classifyImage = async () => {
    if (model && imageUrl) {
      try {
        const imageElement = document.getElementById("uploaded-image");
        const processedImage = preprocessImage(imageElement);
        const predictions = await model.predict(processedImage).data();
        const topPrediction = Array.from(predictions)
          .map((p, i) => ({ probability: p, className: i }))
          .sort((a, b) => b.probability - a.probability)[0];
        const category =
          classNames[topPrediction.className] ||
          `Class ${topPrediction.className}`;
        setPrediction(
          `${category} (${(topPrediction.probability * 100).toFixed(2)}%)`
        );
        tf.dispose(processedImage);
      } catch (error) {
        console.error(`Error during prediction: ${error}`);
        setError("Failed to classify the image. Please try again.");
      }
    }
  };

  return (
    <div className="image-classifier">
      <h1>E-commerce Product Classifier</h1>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {imageUrl && (
        <div>
          <img
            id="uploaded-image"
            src={imageUrl}
            alt="Uploaded"
            onLoad={classifyImage}
          />
          <h2>
            Prediction:{" "}
            {error ? error : prediction ? prediction : "Processing..."}
          </h2>
        </div>
      )}
    </div>
  );
}

export default App;
