# Adobe Hackathon 1B – Docker Setup Guide

This guide helps you build and run the project in an isolated Docker environment using Python 3.10 and predownloaded packages (with fallback to online pip install).

---


```
  docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier
  docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
``` 
