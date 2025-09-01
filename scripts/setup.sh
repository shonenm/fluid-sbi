#!/bin/bash
set -e

echo "🚀 Setting up Fluid SBI (SDA) Docker environment..."

# Check if we're in the correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Please run this script from the project root."
    exit 1
fi

# Initialize and update submodules
echo "📦 Initializing and updating SDA submodule..."
if [ ! -d "sda/.git" ]; then
    git submodule update --init --recursive
else
    echo "📦 SDA submodule already initialized."
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{inputs,outputs,raw,processed}
mkdir -p results/{models,figures,logs}

# Create .gitkeep files to preserve directory structure
touch data/.gitkeep
touch data/inputs/.gitkeep
touch data/outputs/.gitkeep
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch results/.gitkeep
touch results/models/.gitkeep
touch results/figures/.gitkeep
touch results/logs/.gitkeep

# Setup environment file
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your Weights & Biases API key"
    echo "   Get your API key from: https://wandb.ai/settings"
else
    echo "⚙️  .env file already exists."
fi

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.sh

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker-compose build

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file: nano .env"
echo "2. Add your WANDB_API_KEY"
echo "3. Start environment: ./scripts/dev.sh start"
echo "4. Enter container: ./scripts/dev.sh shell"
echo ""
echo "🔧 Available commands:"
echo "  ./scripts/dev.sh start     # Start development environment"
echo "  ./scripts/dev.sh shell     # Enter development container" 
echo "  ./scripts/dev.sh jupyter   # Start Jupyter Lab"
echo "  ./scripts/run-experiments.sh lorenz     # Run Lorenz experiment"
echo "  ./scripts/update-sda.sh    # Update SDA to latest version"
echo ""
echo "📚 For more info, see README.md"