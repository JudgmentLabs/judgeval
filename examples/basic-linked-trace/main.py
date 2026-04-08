from judgeval import Tracer


Tracer.init(project_name="basic-linked-trace")


@Tracer.observe(span_type="tool")
def search_flights(destination: str) -> list[str]:
    return [
        f"Morning flight to {destination}",
        f"Evening flight to {destination}",
    ]


@Tracer.observe(span_type="tool")
def search_hotels(destination: str) -> str:
    return f"Central hotel in {destination}"


@Tracer.observe(span_type="agent", fork=True)
def plan_transport_and_lodging(destination: str) -> dict[str, object]:
    flights = search_flights(destination)
    hotel = search_hotels(destination)
    return {
        "flight": flights[0],
        "hotel": hotel,
    }


@Tracer.observe(span_type="agent")
def build_trip(destination: str) -> dict[str, object]:
    Tracer.set_session_id("trip-session-1")
    Tracer.set_customer_id("customer-123")
    Tracer.set_customer_user_id("user-123")

    logistics = plan_transport_and_lodging(destination)

    return {
        "destination": destination,
        "logistics": logistics,
        "summary": f"Booked {logistics['flight']} and {logistics['hotel']}",
    }


if __name__ == "__main__":
    trip = build_trip("Paris")
    print(trip)
