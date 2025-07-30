# Adobe Hackathon 1B – Docker Setup Guide

This guide helps you build and run the project in an isolated Docker environment using Python 3.10 and predownloaded packages (with fallback to online pip install).

---


```

  docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
  docker run --rm --platform linux/amd64 -v ${PWD}/output.json:/app/output.json mysolutionname:somerandomidentifier python -m main --persona "travel planner" --query "plan a four day trip"
 
``` 
