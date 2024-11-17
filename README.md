# Ollama GPU Calculator

A web application that helps users determine if their GPU meets the VRAM requirements for running various Ollama LLM models. Check it out at [http://aleibovici.github.io/ollama-gpu-calculator/](http://aleibovici.github.io/ollama-gpu-calculator/).

## About

This calculator helps you:
- Check if your GPU has sufficient VRAM for specific Ollama models
- Compare different LLM model requirements
- Determine approximate GPU specifications needed for your desired models
- Optimize model selection based on your available hardware

Join the discussion about this tool on [Reddit](https://www.reddit.com/r/ollama/comments/1gdux20/ollama_gpu_compatibility_calculator/).

## GPU Requirements

Different Ollama models have varying VRAM requirements:
- Smaller models (3B-7B parameters) typically need 4-8GB VRAM
- Medium models (13B parameters) usually require 8-16GB VRAM
- Larger models (30B-65B parameters) need 24GB+ VRAM

Use this calculator to get ballpark estimates for your specific use case.

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Contributing

Feel free to open issues or submit pull requests if you'd like to contribute to this project.

## Learn More

To learn more about Ollama, visit the [official Ollama documentation](https://ollama.ai/docs).

For React documentation, check out the [React documentation](https://reactjs.org/).
