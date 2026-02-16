👩‍💻 Author
Abha Khakurdikar
Founder – PlanWise
Netherlands


🌱 PlanWise Fresh Produce

Enterprise-grade demand planning and forecasting platform for fresh produce supply chains.

PlanWise Fresh Produce is a full-stack application designed to support data-driven forecasting, scenario planning, and data cleansing for multi-level product, channel, and location hierarchies.

🚀 Overview

PlanWise Fresh Produce enables:

📊 Historical demand analysis (Daily / Weekly / Monthly)

🔮 Forecast generation and evaluation

🧼 Data cleansing & profile management

🔎 Advanced search filters (Product / Channel / Location)

🧠 Scenario-ready architecture (coming next)

🗄 PostgreSQL-backed scalable data model

The system is built to support enterprise-scale data with an extensible schema design.

🏗 Architecture
PlanWiseFreshProduce/
│
├── ui_framework/      → Angular frontend (CoreUI-based)
├── ui_backend/        → FastAPI backend (PostgreSQL)
├── db/                → Database schema + dumps
├── Data/              → Local data files (optional)
└── README.md

🧩 Technology Stack
Frontend

Angular (Standalone components)
CoreUI
TypeScript

Backend

FastAPI
PostgreSQL
SQLAlchemy
Psycopg2

Database

PostgreSQL schema: planwise_fresh_produce

📂 Data Model

The database is designed around hierarchical enterprise structures:
Product
ProductID
ProductDescr
Level
BusinessUnit
ProductFamily
IsDailyForecastRequired
IsNew

Channel
ChannelID
ChannelDescr
Level
Location
LocationID
LocationDescr
Level
Geography

History Tables
Daily / Weekly / Monthly
ProductID, ChannelID, LocationID
StartDate, EndDate
Qty, NetPrice
Level

Forecast Tables
Daily / Weekly / Monthly
Method
Type
Period
Qty

🧼 Cleanse Module

Supports:
Cleanse Profiles
Saved Search integration
Rule-based adjustments
Future: automated correction pipelines

🔍 Saved Search

Users can:
Select Product / Channel / Location attributes
Apply AND / OR logic
Save named searches
Reuse searches across modules

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/<your-username>/planwise-fresh-produce.git
cd planwise-fresh-produce

2️⃣ Backend Setup
cd ui_backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Run backend:
uvicorn main:app --reload

Backend runs on:
http://localhost:8000

3️⃣ Frontend Setup
cd ui_framework
npm install
ng serve


Frontend runs on:
http://localhost:4200

🗄 Database Setup
Create database:
createdb planwise

Restore schema:
psql -U <your-user> -d planwise -f db/planwise_fresh_produce_schema.sql

🔐 Environment Variables

Backend expects:
DATABASE_URL=postgresql://user:password@localhost:5432/planwise
Create a .env file in ui_backend/.

🌍 Vision
PlanWise Fresh Produce is designed as a scalable forecasting engine for:
High-SKU fresh produce businesses
Weather-sensitive supply chains
Multi-channel retail operations
Enterprise scenario simulation

Future roadmap includes:
Scenario cloning engine
Weather integration
Promotion modeling
Forecast accuracy dashboard
Automated ML model benchmarking
Role-based access control

📈 Project Status

Active development.
Core modules:
✔ History
✔ Forecast
✔ Saved Search
✔ Cleanse Profiles
🚧 Scenario Manager (next)


