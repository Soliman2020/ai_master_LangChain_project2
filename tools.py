"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager
from dotenv import load_dotenv
load_dotenv()

# Initialize database manager
db_manager = DatabaseManager()

# Cached vectorstore instance
_vectorstore: Optional[Chroma] = None


def _get_embedding_client() -> OpenAIEmbeddings:
    """Get OpenAIEmbeddings client with credential fallback."""
    # Priority: VOCAREUM_API_KEY -> OPENAI_API_KEY
    api_key = (
        os.getenv("VOCAREUM_API_KEY") or
        os.getenv("OPENAI_API_KEY") or
        os.getenv("OPENAI_ADMIN_KEY")
    )

    if not api_key:
        raise ValueError(
            "Missing API credentials. Set one of: VOCAREUM_API_KEY, OPENAI_API_KEY, or OPENAI_ADMIN_KEY"
        )

    return OpenAIEmbeddings(
        base_url="https://openai.vocareum.com/v1",
        api_key=api_key
    )


def _get_vectorstore() -> Chroma:
    """Get or create the vectorstore singleton."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    persist_directory = "data/vectorstore"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    embeddings = _get_embedding_client()

    # Load documents if vector store doesn't exist
    if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        documents = []
        for doc_path in ["data/documents/tip_device_best_practices.txt", "data/documents/tip_energy_savings.txt"]:
            if os.path.exists(doc_path):
                loader = TextLoader(doc_path)
                docs = loader.load()
                documents.extend(docs)

        if not documents:
            raise FileNotFoundError("No documents found in data/documents/")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        _vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        _vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    return _vectorstore


# TODO: Implement get_weather_forecast tool
@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.

    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)

    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
        E.g:
        forecast = {
            "location": ...,
            "forecast_days": ...,
            "current": {
                "temperature_c": ...,
                "condition": random.choice(["sunny", "partly_cloudy", "cloudy"]),
                "humidity": ...,
                "wind_speed": ...
            },
            "hourly": [
                {
                    "hour": ..., # for hour in range(24)
                    "temperature_c": ...,
                    "condition": ...,
                    "solar_irradiance": ...,
                    "humidity": ...,
                    "wind_speed": ...
                },
            ]
        }
    """
    # Clamp days to valid range
    days = max(1, min(7, days))

    conditions = ["sunny", "partly_cloudy", "cloudy", "rainy"]
    base_temp = random.uniform(15, 30)  # Base temperature based on location/season
    base_humidity = random.uniform(40, 70)
    base_wind = random.uniform(5, 20)

    # Determine primary condition for the forecast
    primary_condition = random.choices(
        conditions,
        weights=[0.4, 0.3, 0.2, 0.1]
    )[0]

    current = {
        "temperature_c": round(base_temp, 1),
        "condition": primary_condition,
        "humidity": round(base_humidity, 1),
        "wind_speed": round(base_wind, 1)
    }

    # Generate hourly forecast for the first day
    hourly = []
    for hour in range(24):
        # Temperature varies throughout the day: cooler at night, warmer midday
        temp_variation = -3 if hour < 6 else (5 if 11 <= hour <= 15 else 0)
        hour_temp = base_temp + temp_variation + random.uniform(-2, 2)

        # Solar irradiance peaks midday, zero at night
        if 6 <= hour <= 18:
            solar_irradiance = round(random.uniform(200, 800) * (1 if hour < 12 else 0.8), 1)
        else:
            solar_irradiance = 0

        # Condition may change slightly throughout the day
        hour_condition = primary_condition
        if random.random() < 0.2:
            hour_condition = random.choice(conditions)

        hourly.append({
            "hour": hour,
            "temperature_c": round(hour_temp, 1),
            "condition": hour_condition,
            "solar_irradiance": solar_irradiance,
            "humidity": round(base_humidity + random.uniform(-10, 10), 1),
            "wind_speed": round(base_wind + random.uniform(-5, 5), 1)
        })

    return {
        "location": location,
        "forecast_days": days,
        "current": current,
        "hourly": hourly
    }

# TODO: Implement get_electricity_prices tool
@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates
        E.g:
        prices = {
            "date": ...,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "hourly_rates": [
                {
                    "hour": .., # for hour in range(24)
                    "rate": ..,
                    "period": ..,
                    "demand_charge": ...
                }
            ]
        }
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Time-of-use pricing tiers
    # Off-peak: 22:00 - 6:00 (night)
    # Part-peak: 6:00 - 10:00, 14:00 - 16:00 (morning/afternoon)
    # Peak: 10:00 - 14:00, 16:00 - 22:00 (midday/evening)

    # Base rates (USD per kWh)
    off_peak_rate = 0.08
    part_peak_rate = 0.12
    peak_rate = 0.18

    # Demand charges (only applies during peak hours)
    peak_demand_charge = 0.05

    hourly_rates = []
    for hour in range(24):
        if 22 <= hour or hour < 6:
            # Off-peak: night hours
            period = "off_peak"
            rate = off_peak_rate
            demand_charge = 0
        elif (6 <= hour < 10) or (14 <= hour < 16):
            # Part-peak: morning and early afternoon
            period = "part_peak"
            rate = part_peak_rate
            demand_charge = 0
        else:
            # Peak: midday and evening
            period = "peak"
            rate = peak_rate
            demand_charge = peak_demand_charge

        hourly_rates.append({
            "hour": hour,
            "rate": round(rate, 4),
            "period": period,
            "demand_charge": round(demand_charge, 4)
        })

    return {
        "date": date,
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "hourly_rates": hourly_rates
    }

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")

    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_usage_by_date_range(start_dt, end_dt)

        if device_type:
            records = [r for r in records if r.device_type == device_type]

        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }

        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })

        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_generation_by_date_range(start_dt, end_dt)

        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }

        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "temperature_c": record.temperature_c,
                "solar_irradiance": record.solar_irradiance
            })

        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.

    Args:
        hours (int): Number of hours to look back (default 24)

    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)

        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }

        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1

        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)

        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.

    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        vectorstore = _get_vectorstore()

        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)

        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }

        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })

        return results
    except FileNotFoundError as e:
        return {"error": f"Failed to search energy tips: {e}"}
    except ValueError as e:
        return {"error": f"Failed to search energy tips: {e}"}
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}


@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float,
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.

    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)

    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0

    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]