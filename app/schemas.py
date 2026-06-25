from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class EventTrackRequest(BaseModel):
    userId: str = Field(..., examples=["123"])
    event: str = Field(..., examples=["user viewed pricing page"])
    page: str = Field(..., examples=["/pricing"])
    metadata: Dict[str, Any] = Field(default={})
    timestamp: Optional[datetime] = None


class UserActivity(BaseModel):
    user_id: str
    event_count: int


class AnalyticsResponse(BaseModel):
    total_events: int
    events_per_user: List[UserActivity]
    most_active_users: List[UserActivity]


class SearchResultItem(BaseModel):
    event_id: str
    user_id: str
    event: str
    page: str
    timestamp: datetime
    similarity_score: float


class SimilarUserItem(BaseModel):
    user_id: str
    similarity_score: float


class SimilarUsersResponse(BaseModel):
    target_user_id: str
    similar_users: List[SimilarUserItem]

