CREATE TABLE IF NOT EXISTS planwise_fresh_produce.weather_forecast_daily (
    "LocationID" VARCHAR(255) NOT NULL,
    "Date" DATE NOT NULL,

    "TavgC" DOUBLE PRECISION,
    "TminC" DOUBLE PRECISION,
    "TmaxC" DOUBLE PRECISION,
    "PrecipMM" DOUBLE PRECISION,
    "WindMaxMS" DOUBLE PRECISION,
    "SunHours" DOUBLE PRECISION,

    "Provider" VARCHAR(100),
    "Model" VARCHAR(100),
    "ForecastRun" TIMESTAMP,
    "LoadedAt" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("LocationID", "Date")
);

CREATE INDEX IF NOT EXISTS idx_weather_forecast_daily_date
ON planwise_fresh_produce.weather_forecast_daily ("Date");
