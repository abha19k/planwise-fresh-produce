--
-- PostgreSQL database dump
--

-- Dumped from database version 14.17 (Homebrew)
-- Dumped by pg_dump version 14.17 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: planwise_fresh_produce; Type: SCHEMA; Schema: -; Owner: abha
--

CREATE SCHEMA planwise_fresh_produce;


ALTER SCHEMA planwise_fresh_produce OWNER TO abha;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: channel; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.channel (
    "ChannelID" text,
    "ChannelDescr" text,
    "Level" integer,
    "IsActive" boolean
);


ALTER TABLE planwise_fresh_produce.channel OWNER TO abha;

--
-- Name: classified_forecast_elements; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.classified_forecast_elements (
    "ProductID" text NOT NULL,
    "ChannelID" text NOT NULL,
    "LocationID" text NOT NULL,
    "Period" text NOT NULL,
    "ADI" double precision,
    "CV2" double precision,
    "Category" text NOT NULL,
    "Algorithm" text NOT NULL,
    "CreatedAt" timestamp with time zone DEFAULT now() NOT NULL,
    "UpdatedAt" timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE planwise_fresh_produce.classified_forecast_elements OWNER TO abha;

--
-- Name: cleanse_profiles; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.cleanse_profiles (
    id integer NOT NULL,
    name text NOT NULL,
    config jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE planwise_fresh_produce.cleanse_profiles OWNER TO abha;

--
-- Name: cleanse_profiles_id_seq; Type: SEQUENCE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE SEQUENCE planwise_fresh_produce.cleanse_profiles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE planwise_fresh_produce.cleanse_profiles_id_seq OWNER TO abha;

--
-- Name: cleanse_profiles_id_seq; Type: SEQUENCE OWNED BY; Schema: planwise_fresh_produce; Owner: abha
--

ALTER SEQUENCE planwise_fresh_produce.cleanse_profiles_id_seq OWNED BY planwise_fresh_produce.cleanse_profiles.id;


--
-- Name: forecast_111_daily; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_111_daily (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_111_daily OWNER TO abha;

--
-- Name: forecast_111_daily_baseline; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_111_daily_baseline (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_111_daily_baseline OWNER TO abha;

--
-- Name: forecast_111_monthly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_111_monthly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_111_monthly OWNER TO abha;

--
-- Name: forecast_111_monthly_baseline; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_111_monthly_baseline (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_111_monthly_baseline OWNER TO abha;

--
-- Name: forecast_111_weekly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_111_weekly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_111_weekly OWNER TO abha;

--
-- Name: forecast_111_weekly_baseline; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_111_weekly_baseline (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_111_weekly_baseline OWNER TO abha;

--
-- Name: forecast_121_daily; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_121_daily (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_121_daily OWNER TO abha;

--
-- Name: forecast_121_daily_baseline; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_121_daily_baseline (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_121_daily_baseline OWNER TO abha;

--
-- Name: forecast_121_monthly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_121_monthly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_121_monthly OWNER TO abha;

--
-- Name: forecast_121_monthly_baseline; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_121_monthly_baseline (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_121_monthly_baseline OWNER TO abha;

--
-- Name: forecast_121_weekly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_121_weekly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_121_weekly OWNER TO abha;

--
-- Name: forecast_121_weekly_baseline; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_121_weekly_baseline (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_121_weekly_baseline OWNER TO abha;

--
-- Name: forecast_221_daily; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_221_daily (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_221_daily OWNER TO abha;

--
-- Name: forecast_221_daily_baseline; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_221_daily_baseline (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_221_daily_baseline OWNER TO abha;

--
-- Name: forecast_221_monthly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_221_monthly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_221_monthly OWNER TO abha;

--
-- Name: forecast_221_monthly_baseline; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_221_monthly_baseline (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_221_monthly_baseline OWNER TO abha;

--
-- Name: forecast_221_weekly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_221_weekly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_221_weekly OWNER TO abha;

--
-- Name: forecast_221_weekly_baseline; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecast_221_weekly_baseline (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "StartDate" date,
    "EndDate" date,
    "Period" text,
    "Qty" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision,
    "ForecastQty" double precision,
    "Method" text
);


ALTER TABLE planwise_fresh_produce.forecast_221_weekly_baseline OWNER TO abha;

--
-- Name: forecastelement; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.forecastelement (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Level" integer,
    "IsActive" boolean
);


ALTER TABLE planwise_fresh_produce.forecastelement OWNER TO abha;

--
-- Name: history_111_daily; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_111_daily (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Qty" double precision,
    "Period" text,
    "StartDate" date,
    "EndDate" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision
);


ALTER TABLE planwise_fresh_produce.history_111_daily OWNER TO abha;

--
-- Name: history_111_monthly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_111_monthly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Qty" double precision,
    "Period" text,
    "StartDate" date,
    "EndDate" date,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision
);


ALTER TABLE planwise_fresh_produce.history_111_monthly OWNER TO abha;

--
-- Name: history_111_weekly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_111_weekly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Qty" double precision,
    "Period" text,
    "StartDate" date,
    "EndDate" date,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision
);


ALTER TABLE planwise_fresh_produce.history_111_weekly OWNER TO abha;

--
-- Name: history_121_daily; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_121_daily (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Qty" double precision,
    "Period" text,
    "StartDate" date,
    "EndDate" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision
);


ALTER TABLE planwise_fresh_produce.history_121_daily OWNER TO abha;

--
-- Name: history_121_monthly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_121_monthly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Qty" double precision,
    "Period" text,
    "StartDate" date,
    "EndDate" date,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision
);


ALTER TABLE planwise_fresh_produce.history_121_monthly OWNER TO abha;

--
-- Name: history_121_weekly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_121_weekly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Qty" double precision,
    "Period" text,
    "StartDate" date,
    "EndDate" date,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision
);


ALTER TABLE planwise_fresh_produce.history_121_weekly OWNER TO abha;

--
-- Name: history_221_daily; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_221_daily (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Qty" double precision,
    "Period" text,
    "StartDate" date,
    "EndDate" text,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision
);


ALTER TABLE planwise_fresh_produce.history_221_daily OWNER TO abha;

--
-- Name: history_221_monthly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_221_monthly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Qty" double precision,
    "Period" text,
    "StartDate" date,
    "EndDate" date,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision
);


ALTER TABLE planwise_fresh_produce.history_221_monthly OWNER TO abha;

--
-- Name: history_221_weekly; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_221_weekly (
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "Qty" double precision,
    "Period" text,
    "StartDate" date,
    "EndDate" date,
    "UOM" text,
    "NetPrice" double precision,
    "ListPrice" double precision
);


ALTER TABLE planwise_fresh_produce.history_221_weekly OWNER TO abha;

--
-- Name: history_cleansed; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.history_cleansed (
    id bigint NOT NULL,
    "ProductID" text NOT NULL,
    "ChannelID" text NOT NULL,
    "LocationID" text NOT NULL,
    "StartDate" date NOT NULL,
    "EndDate" date NOT NULL,
    "Period" text NOT NULL,
    "Qty" double precision NOT NULL,
    "Level" text,
    "CreatedAt" timestamp with time zone DEFAULT now() NOT NULL,
    "Type" text DEFAULT 'Cleansed-History'::text NOT NULL,
    "NetPrice" double precision NOT NULL,
    "ListPrice" double precision NOT NULL
);


ALTER TABLE planwise_fresh_produce.history_cleansed OWNER TO abha;

--
-- Name: history_cleansed_id_seq; Type: SEQUENCE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE SEQUENCE planwise_fresh_produce.history_cleansed_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE planwise_fresh_produce.history_cleansed_id_seq OWNER TO abha;

--
-- Name: history_cleansed_id_seq; Type: SEQUENCE OWNED BY; Schema: planwise_fresh_produce; Owner: abha
--

ALTER SEQUENCE planwise_fresh_produce.history_cleansed_id_seq OWNED BY planwise_fresh_produce.history_cleansed.id;


--
-- Name: location; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.location (
    "LocationID" text,
    "LocationDescr" text,
    "Level" integer,
    "IsActive" boolean
);


ALTER TABLE planwise_fresh_produce.location OWNER TO abha;

--
-- Name: product; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.product (
    "ProductID" text,
    "ProductDescr" text,
    "Level" integer,
    "BusinessUnit" text,
    "IsDailyForecastRequired" boolean,
    "IsNew" boolean,
    "ProductFamily" text
);


ALTER TABLE planwise_fresh_produce.product OWNER TO abha;

--
-- Name: promotions; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.promotions (
    "PromoID" text,
    "PromoName" text,
    "StartDate" date,
    "EndDate" date,
    "ProductID" text,
    "ChannelID" text,
    "LocationID" text,
    "PromoLevel" integer,
    "DiscountPct" integer,
    "UpliftPct" integer,
    "Notes" text
);


ALTER TABLE planwise_fresh_produce.promotions OWNER TO abha;

--
-- Name: saved_searches; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.saved_searches (
    id integer NOT NULL,
    name text NOT NULL,
    query text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE planwise_fresh_produce.saved_searches OWNER TO abha;

--
-- Name: saved_searches_id_seq; Type: SEQUENCE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE SEQUENCE planwise_fresh_produce.saved_searches_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE planwise_fresh_produce.saved_searches_id_seq OWNER TO abha;

--
-- Name: saved_searches_id_seq; Type: SEQUENCE OWNED BY; Schema: planwise_fresh_produce; Owner: abha
--

ALTER SEQUENCE planwise_fresh_produce.saved_searches_id_seq OWNED BY planwise_fresh_produce.saved_searches.id;


--
-- Name: v_forecast_baseline; Type: VIEW; Schema: planwise_fresh_produce; Owner: abha
--

CREATE VIEW planwise_fresh_produce.v_forecast_baseline AS
 SELECT '111'::text AS "Level",
    'baseline'::text AS "Model",
    forecast_111_daily_baseline."ProductID",
    forecast_111_daily_baseline."ChannelID",
    forecast_111_daily_baseline."LocationID",
    forecast_111_daily_baseline."StartDate",
    forecast_111_daily_baseline."EndDate",
    forecast_111_daily_baseline."Period",
    forecast_111_daily_baseline."ForecastQty",
    forecast_111_daily_baseline."UOM",
    forecast_111_daily_baseline."NetPrice",
    forecast_111_daily_baseline."ListPrice",
    forecast_111_daily_baseline."Method"
   FROM planwise_fresh_produce.forecast_111_daily_baseline
UNION ALL
 SELECT '121'::text AS "Level",
    'baseline'::text AS "Model",
    forecast_121_daily_baseline."ProductID",
    forecast_121_daily_baseline."ChannelID",
    forecast_121_daily_baseline."LocationID",
    forecast_121_daily_baseline."StartDate",
    forecast_121_daily_baseline."EndDate",
    forecast_121_daily_baseline."Period",
    forecast_121_daily_baseline."ForecastQty",
    forecast_121_daily_baseline."UOM",
    forecast_121_daily_baseline."NetPrice",
    forecast_121_daily_baseline."ListPrice",
    forecast_121_daily_baseline."Method"
   FROM planwise_fresh_produce.forecast_121_daily_baseline
UNION ALL
 SELECT '221'::text AS "Level",
    'baseline'::text AS "Model",
    forecast_221_daily_baseline."ProductID",
    forecast_221_daily_baseline."ChannelID",
    forecast_221_daily_baseline."LocationID",
    forecast_221_daily_baseline."StartDate",
    forecast_221_daily_baseline."EndDate",
    forecast_221_daily_baseline."Period",
    forecast_221_daily_baseline."ForecastQty",
    forecast_221_daily_baseline."UOM",
    forecast_221_daily_baseline."NetPrice",
    forecast_221_daily_baseline."ListPrice",
    forecast_221_daily_baseline."Method"
   FROM planwise_fresh_produce.forecast_221_daily_baseline
UNION ALL
 SELECT '111'::text AS "Level",
    'baseline'::text AS "Model",
    forecast_111_weekly_baseline."ProductID",
    forecast_111_weekly_baseline."ChannelID",
    forecast_111_weekly_baseline."LocationID",
    forecast_111_weekly_baseline."StartDate",
    forecast_111_weekly_baseline."EndDate",
    forecast_111_weekly_baseline."Period",
    forecast_111_weekly_baseline."ForecastQty",
    forecast_111_weekly_baseline."UOM",
    forecast_111_weekly_baseline."NetPrice",
    forecast_111_weekly_baseline."ListPrice",
    forecast_111_weekly_baseline."Method"
   FROM planwise_fresh_produce.forecast_111_weekly_baseline
UNION ALL
 SELECT '121'::text AS "Level",
    'baseline'::text AS "Model",
    forecast_121_weekly_baseline."ProductID",
    forecast_121_weekly_baseline."ChannelID",
    forecast_121_weekly_baseline."LocationID",
    forecast_121_weekly_baseline."StartDate",
    forecast_121_weekly_baseline."EndDate",
    forecast_121_weekly_baseline."Period",
    forecast_121_weekly_baseline."ForecastQty",
    forecast_121_weekly_baseline."UOM",
    forecast_121_weekly_baseline."NetPrice",
    forecast_121_weekly_baseline."ListPrice",
    forecast_121_weekly_baseline."Method"
   FROM planwise_fresh_produce.forecast_121_weekly_baseline
UNION ALL
 SELECT '221'::text AS "Level",
    'baseline'::text AS "Model",
    forecast_221_weekly_baseline."ProductID",
    forecast_221_weekly_baseline."ChannelID",
    forecast_221_weekly_baseline."LocationID",
    forecast_221_weekly_baseline."StartDate",
    forecast_221_weekly_baseline."EndDate",
    forecast_221_weekly_baseline."Period",
    forecast_221_weekly_baseline."ForecastQty",
    forecast_221_weekly_baseline."UOM",
    forecast_221_weekly_baseline."NetPrice",
    forecast_221_weekly_baseline."ListPrice",
    forecast_221_weekly_baseline."Method"
   FROM planwise_fresh_produce.forecast_221_weekly_baseline
UNION ALL
 SELECT '111'::text AS "Level",
    'baseline'::text AS "Model",
    forecast_111_monthly_baseline."ProductID",
    forecast_111_monthly_baseline."ChannelID",
    forecast_111_monthly_baseline."LocationID",
    forecast_111_monthly_baseline."StartDate",
    forecast_111_monthly_baseline."EndDate",
    forecast_111_monthly_baseline."Period",
    forecast_111_monthly_baseline."ForecastQty",
    forecast_111_monthly_baseline."UOM",
    forecast_111_monthly_baseline."NetPrice",
    forecast_111_monthly_baseline."ListPrice",
    forecast_111_monthly_baseline."Method"
   FROM planwise_fresh_produce.forecast_111_monthly_baseline
UNION ALL
 SELECT '121'::text AS "Level",
    'baseline'::text AS "Model",
    forecast_121_monthly_baseline."ProductID",
    forecast_121_monthly_baseline."ChannelID",
    forecast_121_monthly_baseline."LocationID",
    forecast_121_monthly_baseline."StartDate",
    forecast_121_monthly_baseline."EndDate",
    forecast_121_monthly_baseline."Period",
    forecast_121_monthly_baseline."ForecastQty",
    forecast_121_monthly_baseline."UOM",
    forecast_121_monthly_baseline."NetPrice",
    forecast_121_monthly_baseline."ListPrice",
    forecast_121_monthly_baseline."Method"
   FROM planwise_fresh_produce.forecast_121_monthly_baseline
UNION ALL
 SELECT '221'::text AS "Level",
    'baseline'::text AS "Model",
    forecast_221_monthly_baseline."ProductID",
    forecast_221_monthly_baseline."ChannelID",
    forecast_221_monthly_baseline."LocationID",
    forecast_221_monthly_baseline."StartDate",
    forecast_221_monthly_baseline."EndDate",
    forecast_221_monthly_baseline."Period",
    forecast_221_monthly_baseline."ForecastQty",
    forecast_221_monthly_baseline."UOM",
    forecast_221_monthly_baseline."NetPrice",
    forecast_221_monthly_baseline."ListPrice",
    forecast_221_monthly_baseline."Method"
   FROM planwise_fresh_produce.forecast_221_monthly_baseline;


ALTER TABLE planwise_fresh_produce.v_forecast_baseline OWNER TO abha;

--
-- Name: v_forecast_feat; Type: VIEW; Schema: planwise_fresh_produce; Owner: abha
--

CREATE VIEW planwise_fresh_produce.v_forecast_feat AS
 SELECT '111'::text AS "Level",
    'feat'::text AS "Model",
    forecast_111_daily."ProductID",
    forecast_111_daily."ChannelID",
    forecast_111_daily."LocationID",
    forecast_111_daily."StartDate",
    forecast_111_daily."EndDate",
    forecast_111_daily."Period",
    forecast_111_daily."ForecastQty",
    forecast_111_daily."UOM",
    forecast_111_daily."NetPrice",
    forecast_111_daily."ListPrice",
    forecast_111_daily."Method"
   FROM planwise_fresh_produce.forecast_111_daily
UNION ALL
 SELECT '121'::text AS "Level",
    'feat'::text AS "Model",
    forecast_121_daily."ProductID",
    forecast_121_daily."ChannelID",
    forecast_121_daily."LocationID",
    forecast_121_daily."StartDate",
    forecast_121_daily."EndDate",
    forecast_121_daily."Period",
    forecast_121_daily."ForecastQty",
    forecast_121_daily."UOM",
    forecast_121_daily."NetPrice",
    forecast_121_daily."ListPrice",
    forecast_121_daily."Method"
   FROM planwise_fresh_produce.forecast_121_daily
UNION ALL
 SELECT '221'::text AS "Level",
    'feat'::text AS "Model",
    forecast_221_daily."ProductID",
    forecast_221_daily."ChannelID",
    forecast_221_daily."LocationID",
    forecast_221_daily."StartDate",
    forecast_221_daily."EndDate",
    forecast_221_daily."Period",
    forecast_221_daily."ForecastQty",
    forecast_221_daily."UOM",
    forecast_221_daily."NetPrice",
    forecast_221_daily."ListPrice",
    forecast_221_daily."Method"
   FROM planwise_fresh_produce.forecast_221_daily
UNION ALL
 SELECT '111'::text AS "Level",
    'feat'::text AS "Model",
    forecast_111_weekly."ProductID",
    forecast_111_weekly."ChannelID",
    forecast_111_weekly."LocationID",
    forecast_111_weekly."StartDate",
    forecast_111_weekly."EndDate",
    forecast_111_weekly."Period",
    forecast_111_weekly."ForecastQty",
    forecast_111_weekly."UOM",
    forecast_111_weekly."NetPrice",
    forecast_111_weekly."ListPrice",
    forecast_111_weekly."Method"
   FROM planwise_fresh_produce.forecast_111_weekly
UNION ALL
 SELECT '121'::text AS "Level",
    'feat'::text AS "Model",
    forecast_121_weekly."ProductID",
    forecast_121_weekly."ChannelID",
    forecast_121_weekly."LocationID",
    forecast_121_weekly."StartDate",
    forecast_121_weekly."EndDate",
    forecast_121_weekly."Period",
    forecast_121_weekly."ForecastQty",
    forecast_121_weekly."UOM",
    forecast_121_weekly."NetPrice",
    forecast_121_weekly."ListPrice",
    forecast_121_weekly."Method"
   FROM planwise_fresh_produce.forecast_121_weekly
UNION ALL
 SELECT '221'::text AS "Level",
    'feat'::text AS "Model",
    forecast_221_weekly."ProductID",
    forecast_221_weekly."ChannelID",
    forecast_221_weekly."LocationID",
    forecast_221_weekly."StartDate",
    forecast_221_weekly."EndDate",
    forecast_221_weekly."Period",
    forecast_221_weekly."ForecastQty",
    forecast_221_weekly."UOM",
    forecast_221_weekly."NetPrice",
    forecast_221_weekly."ListPrice",
    forecast_221_weekly."Method"
   FROM planwise_fresh_produce.forecast_221_weekly
UNION ALL
 SELECT '111'::text AS "Level",
    'feat'::text AS "Model",
    forecast_111_monthly."ProductID",
    forecast_111_monthly."ChannelID",
    forecast_111_monthly."LocationID",
    forecast_111_monthly."StartDate",
    forecast_111_monthly."EndDate",
    forecast_111_monthly."Period",
    forecast_111_monthly."ForecastQty",
    forecast_111_monthly."UOM",
    forecast_111_monthly."NetPrice",
    forecast_111_monthly."ListPrice",
    forecast_111_monthly."Method"
   FROM planwise_fresh_produce.forecast_111_monthly
UNION ALL
 SELECT '121'::text AS "Level",
    'feat'::text AS "Model",
    forecast_121_monthly."ProductID",
    forecast_121_monthly."ChannelID",
    forecast_121_monthly."LocationID",
    forecast_121_monthly."StartDate",
    forecast_121_monthly."EndDate",
    forecast_121_monthly."Period",
    forecast_121_monthly."ForecastQty",
    forecast_121_monthly."UOM",
    forecast_121_monthly."NetPrice",
    forecast_121_monthly."ListPrice",
    forecast_121_monthly."Method"
   FROM planwise_fresh_produce.forecast_121_monthly
UNION ALL
 SELECT '221'::text AS "Level",
    'feat'::text AS "Model",
    forecast_221_monthly."ProductID",
    forecast_221_monthly."ChannelID",
    forecast_221_monthly."LocationID",
    forecast_221_monthly."StartDate",
    forecast_221_monthly."EndDate",
    forecast_221_monthly."Period",
    forecast_221_monthly."ForecastQty",
    forecast_221_monthly."UOM",
    forecast_221_monthly."NetPrice",
    forecast_221_monthly."ListPrice",
    forecast_221_monthly."Method"
   FROM planwise_fresh_produce.forecast_221_monthly;


ALTER TABLE planwise_fresh_produce.v_forecast_feat OWNER TO abha;

--
-- Name: v_history; Type: VIEW; Schema: planwise_fresh_produce; Owner: abha
--

CREATE VIEW planwise_fresh_produce.v_history AS
 SELECT '111'::text AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    COALESCE((NULLIF(h."EndDate", 'None'::text))::date, h."StartDate") AS "EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    'Normal-History'::text AS "Type"
   FROM planwise_fresh_produce.history_111_daily h
UNION ALL
 SELECT '111'::text AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    h."EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    'Normal-History'::text AS "Type"
   FROM planwise_fresh_produce.history_111_weekly h
UNION ALL
 SELECT '111'::text AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    h."EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    'Normal-History'::text AS "Type"
   FROM planwise_fresh_produce.history_111_monthly h
UNION ALL
 SELECT '121'::text AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    COALESCE((NULLIF(h."EndDate", 'None'::text))::date, h."StartDate") AS "EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    'Normal-History'::text AS "Type"
   FROM planwise_fresh_produce.history_121_daily h
UNION ALL
 SELECT '121'::text AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    h."EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    'Normal-History'::text AS "Type"
   FROM planwise_fresh_produce.history_121_weekly h
UNION ALL
 SELECT '121'::text AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    h."EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    'Normal-History'::text AS "Type"
   FROM planwise_fresh_produce.history_121_monthly h
UNION ALL
 SELECT '221'::text AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    COALESCE((NULLIF(h."EndDate", 'None'::text))::date, h."StartDate") AS "EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    'Normal-History'::text AS "Type"
   FROM planwise_fresh_produce.history_221_daily h
UNION ALL
 SELECT '221'::text AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    h."EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    'Normal-History'::text AS "Type"
   FROM planwise_fresh_produce.history_221_weekly h
UNION ALL
 SELECT '221'::text AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    h."EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    'Normal-History'::text AS "Type"
   FROM planwise_fresh_produce.history_221_monthly h
UNION ALL
 SELECT COALESCE(h."Level", ''::text) AS "Level",
    h."ProductID",
    h."ChannelID",
    h."LocationID",
    h."StartDate",
    h."EndDate",
    h."Period",
    h."Qty",
    h."NetPrice",
    h."ListPrice",
    h."Type"
   FROM planwise_fresh_produce.history_cleansed h;


ALTER TABLE planwise_fresh_produce.v_history OWNER TO abha;

--
-- Name: weather_daily; Type: TABLE; Schema: planwise_fresh_produce; Owner: abha
--

CREATE TABLE planwise_fresh_produce.weather_daily (
    "LocationID" text,
    "Date" date,
    "TavgC" double precision,
    "TminC" double precision,
    "TmaxC" double precision,
    "PrecipMM" double precision,
    "WindMaxMS" double precision,
    "SunHours" double precision
);


ALTER TABLE planwise_fresh_produce.weather_daily OWNER TO abha;

--
-- Name: cleanse_profiles id; Type: DEFAULT; Schema: planwise_fresh_produce; Owner: abha
--

ALTER TABLE ONLY planwise_fresh_produce.cleanse_profiles ALTER COLUMN id SET DEFAULT nextval('planwise_fresh_produce.cleanse_profiles_id_seq'::regclass);


--
-- Name: history_cleansed id; Type: DEFAULT; Schema: planwise_fresh_produce; Owner: abha
--

ALTER TABLE ONLY planwise_fresh_produce.history_cleansed ALTER COLUMN id SET DEFAULT nextval('planwise_fresh_produce.history_cleansed_id_seq'::regclass);


--
-- Name: saved_searches id; Type: DEFAULT; Schema: planwise_fresh_produce; Owner: abha
--

ALTER TABLE ONLY planwise_fresh_produce.saved_searches ALTER COLUMN id SET DEFAULT nextval('planwise_fresh_produce.saved_searches_id_seq'::regclass);


--
-- Name: classified_forecast_elements classified_forecast_elements_pkey; Type: CONSTRAINT; Schema: planwise_fresh_produce; Owner: abha
--

ALTER TABLE ONLY planwise_fresh_produce.classified_forecast_elements
    ADD CONSTRAINT classified_forecast_elements_pkey PRIMARY KEY ("ProductID", "ChannelID", "LocationID", "Period");


--
-- Name: cleanse_profiles cleanse_profiles_name_key; Type: CONSTRAINT; Schema: planwise_fresh_produce; Owner: abha
--

ALTER TABLE ONLY planwise_fresh_produce.cleanse_profiles
    ADD CONSTRAINT cleanse_profiles_name_key UNIQUE (name);


--
-- Name: cleanse_profiles cleanse_profiles_pkey; Type: CONSTRAINT; Schema: planwise_fresh_produce; Owner: abha
--

ALTER TABLE ONLY planwise_fresh_produce.cleanse_profiles
    ADD CONSTRAINT cleanse_profiles_pkey PRIMARY KEY (id);


--
-- Name: history_cleansed history_cleansed_ProductID_ChannelID_LocationID_StartDate_P_key; Type: CONSTRAINT; Schema: planwise_fresh_produce; Owner: abha
--

ALTER TABLE ONLY planwise_fresh_produce.history_cleansed
    ADD CONSTRAINT "history_cleansed_ProductID_ChannelID_LocationID_StartDate_P_key" UNIQUE ("ProductID", "ChannelID", "LocationID", "StartDate", "Period");


--
-- Name: history_cleansed history_cleansed_pkey; Type: CONSTRAINT; Schema: planwise_fresh_produce; Owner: abha
--

ALTER TABLE ONLY planwise_fresh_produce.history_cleansed
    ADD CONSTRAINT history_cleansed_pkey PRIMARY KEY (id);


--
-- Name: saved_searches saved_searches_pkey; Type: CONSTRAINT; Schema: planwise_fresh_produce; Owner: abha
--

ALTER TABLE ONLY planwise_fresh_produce.saved_searches
    ADD CONSTRAINT saved_searches_pkey PRIMARY KEY (id);


--
-- Name: idx_history_cleansed_keys; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX idx_history_cleansed_keys ON planwise_fresh_produce.history_cleansed USING btree ("ProductID", "ChannelID", "LocationID", "Period");


--
-- Name: ix_f111d_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_f111d_keys_time ON planwise_fresh_produce.forecast_111_daily USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_f111m_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_f111m_keys_time ON planwise_fresh_produce.forecast_111_monthly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_f111w_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_f111w_keys_time ON planwise_fresh_produce.forecast_111_weekly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_f121d_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_f121d_keys_time ON planwise_fresh_produce.forecast_121_daily USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_f121m_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_f121m_keys_time ON planwise_fresh_produce.forecast_121_monthly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_f121w_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_f121w_keys_time ON planwise_fresh_produce.forecast_121_weekly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_f221d_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_f221d_keys_time ON planwise_fresh_produce.forecast_221_daily USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_f221m_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_f221m_keys_time ON planwise_fresh_produce.forecast_221_monthly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_f221w_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_f221w_keys_time ON planwise_fresh_produce.forecast_221_weekly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_forecast_111_daily_baseline_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_daily_baseline_pcl ON planwise_fresh_produce.forecast_111_daily_baseline USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_111_daily_baseline_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_daily_baseline_startdate ON planwise_fresh_produce.forecast_111_daily_baseline USING btree ("StartDate");


--
-- Name: ix_forecast_111_daily_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_daily_pcl ON planwise_fresh_produce.forecast_111_daily USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_111_daily_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_daily_startdate ON planwise_fresh_produce.forecast_111_daily USING btree ("StartDate");


--
-- Name: ix_forecast_111_monthly_baseline_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_monthly_baseline_pcl ON planwise_fresh_produce.forecast_111_monthly_baseline USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_111_monthly_baseline_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_monthly_baseline_startdate ON planwise_fresh_produce.forecast_111_monthly_baseline USING btree ("StartDate");


--
-- Name: ix_forecast_111_monthly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_monthly_pcl ON planwise_fresh_produce.forecast_111_monthly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_111_monthly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_monthly_startdate ON planwise_fresh_produce.forecast_111_monthly USING btree ("StartDate");


--
-- Name: ix_forecast_111_weekly_baseline_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_weekly_baseline_pcl ON planwise_fresh_produce.forecast_111_weekly_baseline USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_111_weekly_baseline_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_weekly_baseline_startdate ON planwise_fresh_produce.forecast_111_weekly_baseline USING btree ("StartDate");


--
-- Name: ix_forecast_111_weekly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_weekly_pcl ON planwise_fresh_produce.forecast_111_weekly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_111_weekly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_111_weekly_startdate ON planwise_fresh_produce.forecast_111_weekly USING btree ("StartDate");


--
-- Name: ix_forecast_121_daily_baseline_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_daily_baseline_pcl ON planwise_fresh_produce.forecast_121_daily_baseline USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_121_daily_baseline_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_daily_baseline_startdate ON planwise_fresh_produce.forecast_121_daily_baseline USING btree ("StartDate");


--
-- Name: ix_forecast_121_daily_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_daily_pcl ON planwise_fresh_produce.forecast_121_daily USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_121_daily_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_daily_startdate ON planwise_fresh_produce.forecast_121_daily USING btree ("StartDate");


--
-- Name: ix_forecast_121_monthly_baseline_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_monthly_baseline_pcl ON planwise_fresh_produce.forecast_121_monthly_baseline USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_121_monthly_baseline_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_monthly_baseline_startdate ON planwise_fresh_produce.forecast_121_monthly_baseline USING btree ("StartDate");


--
-- Name: ix_forecast_121_monthly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_monthly_pcl ON planwise_fresh_produce.forecast_121_monthly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_121_monthly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_monthly_startdate ON planwise_fresh_produce.forecast_121_monthly USING btree ("StartDate");


--
-- Name: ix_forecast_121_weekly_baseline_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_weekly_baseline_pcl ON planwise_fresh_produce.forecast_121_weekly_baseline USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_121_weekly_baseline_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_weekly_baseline_startdate ON planwise_fresh_produce.forecast_121_weekly_baseline USING btree ("StartDate");


--
-- Name: ix_forecast_121_weekly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_weekly_pcl ON planwise_fresh_produce.forecast_121_weekly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_121_weekly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_121_weekly_startdate ON planwise_fresh_produce.forecast_121_weekly USING btree ("StartDate");


--
-- Name: ix_forecast_221_daily_baseline_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_daily_baseline_pcl ON planwise_fresh_produce.forecast_221_daily_baseline USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_221_daily_baseline_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_daily_baseline_startdate ON planwise_fresh_produce.forecast_221_daily_baseline USING btree ("StartDate");


--
-- Name: ix_forecast_221_daily_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_daily_pcl ON planwise_fresh_produce.forecast_221_daily USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_221_daily_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_daily_startdate ON planwise_fresh_produce.forecast_221_daily USING btree ("StartDate");


--
-- Name: ix_forecast_221_monthly_baseline_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_monthly_baseline_pcl ON planwise_fresh_produce.forecast_221_monthly_baseline USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_221_monthly_baseline_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_monthly_baseline_startdate ON planwise_fresh_produce.forecast_221_monthly_baseline USING btree ("StartDate");


--
-- Name: ix_forecast_221_monthly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_monthly_pcl ON planwise_fresh_produce.forecast_221_monthly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_221_monthly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_monthly_startdate ON planwise_fresh_produce.forecast_221_monthly USING btree ("StartDate");


--
-- Name: ix_forecast_221_weekly_baseline_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_weekly_baseline_pcl ON planwise_fresh_produce.forecast_221_weekly_baseline USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_221_weekly_baseline_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_weekly_baseline_startdate ON planwise_fresh_produce.forecast_221_weekly_baseline USING btree ("StartDate");


--
-- Name: ix_forecast_221_weekly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_weekly_pcl ON planwise_fresh_produce.forecast_221_weekly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_forecast_221_weekly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecast_221_weekly_startdate ON planwise_fresh_produce.forecast_221_weekly USING btree ("StartDate");


--
-- Name: ix_forecastelement_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_forecastelement_pcl ON planwise_fresh_produce.forecastelement USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_h111d_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_h111d_keys_time ON planwise_fresh_produce.history_111_daily USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_h111m_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_h111m_keys_time ON planwise_fresh_produce.history_111_monthly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_h111w_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_h111w_keys_time ON planwise_fresh_produce.history_111_weekly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_h121d_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_h121d_keys_time ON planwise_fresh_produce.history_121_daily USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_h121m_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_h121m_keys_time ON planwise_fresh_produce.history_121_monthly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_h121w_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_h121w_keys_time ON planwise_fresh_produce.history_121_weekly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_h221d_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_h221d_keys_time ON planwise_fresh_produce.history_221_daily USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_h221m_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_h221m_keys_time ON planwise_fresh_produce.history_221_monthly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_h221w_keys_time; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_h221w_keys_time ON planwise_fresh_produce.history_221_weekly USING btree ("ProductID", "ChannelID", "LocationID", "StartDate");


--
-- Name: ix_history_111_daily_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_111_daily_pcl ON planwise_fresh_produce.history_111_daily USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_history_111_daily_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_111_daily_startdate ON planwise_fresh_produce.history_111_daily USING btree ("StartDate");


--
-- Name: ix_history_111_monthly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_111_monthly_pcl ON planwise_fresh_produce.history_111_monthly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_history_111_monthly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_111_monthly_startdate ON planwise_fresh_produce.history_111_monthly USING btree ("StartDate");


--
-- Name: ix_history_111_weekly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_111_weekly_pcl ON planwise_fresh_produce.history_111_weekly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_history_111_weekly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_111_weekly_startdate ON planwise_fresh_produce.history_111_weekly USING btree ("StartDate");


--
-- Name: ix_history_121_daily_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_121_daily_pcl ON planwise_fresh_produce.history_121_daily USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_history_121_daily_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_121_daily_startdate ON planwise_fresh_produce.history_121_daily USING btree ("StartDate");


--
-- Name: ix_history_121_monthly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_121_monthly_pcl ON planwise_fresh_produce.history_121_monthly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_history_121_monthly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_121_monthly_startdate ON planwise_fresh_produce.history_121_monthly USING btree ("StartDate");


--
-- Name: ix_history_121_weekly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_121_weekly_pcl ON planwise_fresh_produce.history_121_weekly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_history_121_weekly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_121_weekly_startdate ON planwise_fresh_produce.history_121_weekly USING btree ("StartDate");


--
-- Name: ix_history_221_daily_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_221_daily_pcl ON planwise_fresh_produce.history_221_daily USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_history_221_daily_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_221_daily_startdate ON planwise_fresh_produce.history_221_daily USING btree ("StartDate");


--
-- Name: ix_history_221_monthly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_221_monthly_pcl ON planwise_fresh_produce.history_221_monthly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_history_221_monthly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_221_monthly_startdate ON planwise_fresh_produce.history_221_monthly USING btree ("StartDate");


--
-- Name: ix_history_221_weekly_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_221_weekly_pcl ON planwise_fresh_produce.history_221_weekly USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_history_221_weekly_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_history_221_weekly_startdate ON planwise_fresh_produce.history_221_weekly USING btree ("StartDate");


--
-- Name: ix_promos_keys_date; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_promos_keys_date ON planwise_fresh_produce.promotions USING btree ("ProductID", "ChannelID", "LocationID", "StartDate", "EndDate");


--
-- Name: ix_promotions_pcl; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_promotions_pcl ON planwise_fresh_produce.promotions USING btree ("ProductID", "ChannelID", "LocationID");


--
-- Name: ix_promotions_startdate; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_promotions_startdate ON planwise_fresh_produce.promotions USING btree ("StartDate");


--
-- Name: ix_weather_daily_date; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_weather_daily_date ON planwise_fresh_produce.weather_daily USING btree ("Date");


--
-- Name: ix_weather_loc_date; Type: INDEX; Schema: planwise_fresh_produce; Owner: abha
--

CREATE INDEX ix_weather_loc_date ON planwise_fresh_produce.weather_daily USING btree ("LocationID", "Date");


--
-- PostgreSQL database dump complete
--

