# Deployment Guide - Heart Disease ML Classification

This guide covers deploying the Heart Disease ML Classification application to cloud platforms and production environments.

## Quick Deployment Options

### Option 1: Streamlit Cloud (Recommended for Quick Demo)

**Advantages**: Free, easy, no DevOps knowledge needed

1. **Create Streamlit Account**
   - Visit https://share.streamlit.io
   - Sign in with GitHub account

2. **Deploy from GitHub**
   - Click "New app"
   - Select your GitHub repo: `KanakAcharya/Heart-Disease-ML-Classification`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Share Live Link**
   - Your app will be live at: `https://share.streamlit.io/KanakAcharya/Heart-Disease-ML-Classification/main/app.py`
   - Add link to README.md

### Option 2: Docker + Heroku (Production-Grade)

**Advantages**: Professional, scalable, industry-standard

#### Prerequisites
```bash
# Install Docker, Heroku CLI
brew install docker heroku/brew/heroku  # macOS
sudo apt install docker.io heroku       # Linux
choco install docker heroku-cli         # Windows
```

#### Deployment Steps

1. **Create Heroku App**
```bash
heroku login
heroku create heart-disease-classifier  # Replace with unique name
```

2. **Configure Dockerfile** (Already provided)
The repo includes a `Dockerfile` configured for Flask/Streamlit. Ensure it has:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

3. **Create `Procfile`**
```bash
echo "web: python app.py" > Procfile
git add Procfile
git commit -m "Add Procfile for Heroku"
```

4. **Deploy**
```bash
# Using Heroku Container Registry (Recommended)
heroku container:login
heroku container:push web
heroku container:release web

# OR using Git (if buildpack configured)
git push heroku main
```

5. **View Logs**
```bash
heroku logs --tail
heroku open
```

6. **Share Public URL**
   - App available at: `https://heart-disease-classifier.herokuapp.com`
   - Add to README:
   ```markdown
   **🌐 Live Demo**: [Heart Disease Classifier](https://heart-disease-classifier.herokuapp.com)
   ```

### Option 3: Docker Compose (Local/VPS)

**Advantages**: Full control, works on any server

```bash
# Build and run locally
docker-compose up --build

# Access at http://localhost:5000

# Deploy to VPS (DigitalOcean, AWS EC2, etc.)
scp -r . user@your-vps:/app
ssh user@your-vps
cd /app && docker-compose up -d
```

### Option 4: Cloud Platforms

#### Google Cloud Run
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud builds submit --tag gcr.io/YOUR_PROJECT/heart-disease
gcloud run deploy heart-disease --image gcr.io/YOUR_PROJECT/heart-disease --platform managed --region us-central1 --allow-unauthenticated
```

#### AWS EC2
```bash
# Launch Ubuntu 20.04 instance
# SSH into instance
sudo apt update && sudo apt install docker.io
git clone https://github.com/KanakAcharya/Heart-Disease-ML-Classification.git
cd Heart-Disease-ML-Classification
sudo docker build -t heart-disease .
sudo docker run -p 5000:5000 heart-disease
```

#### Azure App Service
```bash
az group create --name MyResourceGroup --location eastus
az appservice plan create --name MyServicePlan --resource-group MyResourceGroup --sku B1 --is-linux
az webapp create --resource-group MyResourceGroup --plan MyServicePlan --name heart-disease-app --runtime "python|3.9"
az webapp deployment source config-zip --resource-group MyResourceGroup --name heart-disease-app --src .\repo.zip
```

## GitHub Actions Deployment

Create `.github/workflows/deploy.yml` for automatic deployment on push:

```yaml
name: Deploy to Heroku
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Heroku
        uses: AkhileshNS/heroku-deploy@v3.12.12
        with:
          heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
          heroku_app_name: "heart-disease-classifier"
          heroku_email: "your-email@example.com"
```

## Environment Variables

Set these in your deployment platform:

```bash
FLASK_ENV=production
DEBUG=False
PYTHON_ENV=3.9
PORT=5000
```

## Post-Deployment

1. **Test the Live App**
   - Visit your live URL
   - Test with sample patient data
   - Verify predictions work correctly

2. **Update README**
   - Add live demo link
   - Include deployment status badge

3. **Monitor Performance**
   - Check logs for errors
   - Monitor response times
   - Track usage patterns

4. **Continuous Improvement**
   - Gather user feedback
   - Retrain models with new data
   - Deploy updated versions

## Cost Estimates

| Platform | Cost | Best For |
|----------|------|----------|
| Streamlit Cloud | Free | Demo, portfolio |
| Heroku Free | Free (sleep after 30 mins) | Learning, prototyping |
| Heroku Paid | $7-500+/month | Production |
| AWS EC2 | $5-100+/month | Scalable, professional |
| Google Cloud | Free tier + pay-as-go | Enterprise, high volume |

## Troubleshooting

**App crashes on startup**
- Check `requirements.txt` has all dependencies
- Verify `app.py` path is correct
- Review logs: `heroku logs --tail`

**Models not loading**
- Ensure `.pkl` files are in repo
- Check file paths in `app.py`
- Verify permissions

**Slow predictions**
- Optimize model loading (cache in memory)
- Use model compression
- Upgrade server resources

## Security Best Practices

- Use environment variables for secrets
- Enable HTTPS (auto on most platforms)
- Validate user inputs
- Rate limit API endpoints
- Monitor for unauthorized access

## Next Steps

1. Choose deployment platform
2. Follow platform-specific instructions above
3. Test thoroughly before production
4. Add deployment link to GitHub README
5. Share portfolio with recruiters!

---

**Need help?** See [CONTRIBUTING.md](CONTRIBUTING.md) or open an issue!
