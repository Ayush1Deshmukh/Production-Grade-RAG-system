#!/bin/bash
echo "Linking to Hugging Face..."
git remote add hf https://huggingface.co/spaces/Ayush707/rag-backend

echo "Renaming branch to main..."
git branch -M main

echo "Pushing code to Hugging Face..."
git push -u hf main --force
echo "Done!"
