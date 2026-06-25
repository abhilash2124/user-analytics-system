from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from datetime import datetime
from typing import Optional, List

from app.database import get_db, Base, engine
from app.models import User, Event
from app.schemas import (
    EventTrackRequest,
    AnalyticsResponse,
    SearchResultItem,
    UserActivity,
    SimilarUsersResponse,
    SimilarUserItem,
)
from app.ai_service import generate_embedding

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="User Analytics & Semantic Search Engine",
    description="Backend assignment managing real-time activity metrics alongside vectorized context matching.",
)


@app.post("/track", status_code=201)
def track_event(payload: EventTrackRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.userId).first()
    if not user:
        user = User(id=payload.userId)
        db.add(user)
        db.flush()

    vector_embedding = generate_embedding(payload.event)

    new_event = Event(
        user_id=payload.userId,
        event_type=payload.event,
        page=payload.page,
        metadata_json=payload.metadata,
        timestamp=payload.timestamp or datetime.utcnow(),
        embedding=vector_embedding,
    )
    db.add(new_event)
    db.commit()

    return {
        "status": "success",
        "message": "Event logged and contextualized successfully.",
    }


@app.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(
    event: Optional[str] = None,
    from_date: Optional[datetime] = Query(None, alias="from"),
    to_date: Optional[datetime] = Query(None, alias="to"),
    db: Session = Depends(get_db),
):
    base_query = db.query(Event)
    if event:
        base_query = base_query.filter(Event.event_type == event)
    if from_date:
        base_query = base_query.filter(Event.timestamp >= from_date)
    if to_date:
        base_query = base_query.filter(Event.timestamp <= to_date)

    total_events = base_query.count()

    per_user_data = (
        base_query.with_entities(Event.user_id, func.count(Event.id).label("cnt"))
        .group_by(Event.user_id)
        .all()
    )
    events_per_user = [
        UserActivity(user_id=row[0], event_count=row[1]) for row in per_user_data
    ]

    most_active_data = (
        base_query.with_entities(Event.user_id, func.count(Event.id).label("cnt"))
        .group_by(Event.user_id)
        .order_by(func.count(Event.id).desc())
        .limit(5)
        .all()
    )
    most_active_users = [
        UserActivity(user_id=row[0], event_count=row[1]) for row in most_active_data
    ]

    return AnalyticsResponse(
        total_events=total_events,
        events_per_user=events_per_user,
        most_active_users=most_active_users,
    )


@app.get("/search", response_model=List[SearchResultItem])
def semantic_search(query: str, limit: int = 5, db: Session = Depends(get_db)):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter cannot be blank.")

    query_vector = generate_embedding(query)

    results = (
        db.query(
            Event,
            (1 - Event.embedding.cosine_distance(query_vector)).label("similarity"),
        )
        .order_by(Event.embedding.cosine_distance(query_vector))
        .limit(limit)
        .all()
    )

    search_results = []
    for item, score in results:
        search_results.append(
            SearchResultItem(
                event_id=str(item.id),
                user_id=item.user_id,
                event=item.event_type,
                page=item.page,
                timestamp=item.timestamp,
                similarity_score=float(score),
            )
        )

    return search_results


# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the User Analytics & Semantic Search Engine"}


@app.get("/similar-users", response_model=SimilarUsersResponse)
def get_similar_users(userId: str, limit: int = 5, db: Session = Depends(get_db)):
    # 1. Fetch all event embeddings for the target user
    target_events = db.query(Event.embedding).filter(Event.user_id == userId).all()
    
    if not target_events:
        raise HTTPException(
            status_code=404, 
            detail="Target user profile or event history not found."
        )
        
    # 2. Extract arrays and compute the aggregate mean vector fingerprint
    embeddings = [list(row[0]) for row in target_events if row[0] is not None]
    if not embeddings:
        raise HTTPException(
            status_code=400, 
            detail="No vector footprints found for this user profile."
        )
        
    dimensions = len(embeddings[0])
    avg_vector = [float(sum(col) / len(embeddings)) for col in zip(*embeddings)]

    # 3. Use raw SQL math execution to group other users by their average vector similarity
    # Bypasses comparing row-by-row to find overall behavioral closeness
    query = """
        with user_fingerprints as (
            select user_id, avg(embedding) as avg_emb
            from events
            where user_id != :target_uid
            group by user_id
        )
        select user_id, (1 - (avg_emb <=> CAST(:target_vector AS vector))) as similarity
        from user_fingerprints
        order by avg_emb <=> CAST(:target_vector AS vector)
        limit :row_limit;
    """
    
    raw_results = db.execute(
        text(query), 
        {
            "target_uid": userId, 
            "target_vector": str(avg_vector), 
            "row_limit": limit
        }
    ).all()

    # 4. Map records out cleanly
    similar_list = [
        SimilarUserItem(user_id=row[0], similarity_score=float(row[1])) 
        for row in raw_results
    ]

    return SimilarUsersResponse(
        target_user_id=userId,
        similar_users=similar_list
    )

