
from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Tracker, FormValidationAction, Action
from rasa_sdk.executor import CollectingDispatcher
from datetime import datetime  
from rasa_sdk.events import SlotSet, FollowupAction, AllSlotsReset


class ActionCheckOpenDay(Action):

    def name(self) -> Text:
        return "action_check_open_day"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get user message
        user_message = tracker.latest_message.get('text', '').lower()

        # Extract day name from message (simple example)
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        open_days = ["monday", "tuesday", "wednesday", "thursday", "friday"]

        for day in days:
            if day in user_message:
                if day in open_days:
                    dispatcher.utter_message(text=f"Yes, we’re open on {day.capitalize()}. Would you like to book an appointment?")
                else:
                    dispatcher.utter_message(text=f"No, we’re closed on {day.capitalize()}.")
                return []

        # Fallback response if day not found
        dispatcher.utter_message(text="Could you please tell me which day you want to visit?")
        return []