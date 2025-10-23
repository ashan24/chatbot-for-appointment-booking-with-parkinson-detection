import psycopg2
from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Tracker, FormValidationAction, Action
from rasa_sdk.executor import CollectingDispatcher
from datetime import datetime  
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import SlotSet, FollowupAction, AllSlotsReset
from rasa_sdk.types import DomainDict

DB_PARAMS = {
    'dbname': 'chatbot',
    'user': 'postgres',
    'password': 'auckland',
    'host': 'localhost',
    'port': '5432'
}

class ValidateAppointmentForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_appointment_form"
    def validate_full_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        if isinstance(slot_value, str) and len(slot_value.split()) >= 2:
            return {"full_name": slot_value}
        else:
            dispatcher.utter_message(text="Please enter your full name (first and last name).")
            return {"full_name": None}

    async def validate_date_of_birth(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        full_name = tracker.get_slot("full_name")
        dob = slot_value

        if full_name and dob:
            user_id = get_or_create_user(full_name, dob)
            if user_id:
                
                return {"user_id": user_id}
            else:
                dispatcher.utter_message("Failed to access user ID.")
                return {"user_id": None}

        return {"user_id": None}
    

    def validate_appointment_date(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        try:
            appointment_date = datetime.strptime(slot_value, "%Y-%m-%d")
            weekday = appointment_date.weekday()

            if weekday >= 5:
                dispatcher.utter_message(text="Sorry, we are closed on weekends. Please choose a weekday for your appointment.")
                return {"appointment_date": None}
            
            # Return valid date - let the form handle the confirmation question
            return {"appointment_date": slot_value}
            
        except ValueError:
            dispatcher.utter_message(text="The date format should be YYYY-MM-DD. Please enter a valid date.")
            return {"appointment_date": None}
        

    def validate_appointment_time(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        allowed_times = ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]

        if slot_value in allowed_times:
            return {"appointment_time": slot_value}
        else:
            dispatcher.utter_message(text="Sorry, the available time slots are: 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, or 16:00. Please enter a valid time.")
            return {"appointment_time": None}

class Validatedoctoravailabilityform(FormValidationAction):
    def name(self) -> Text:
        return "validate_doctoravailability_form"
    def validate_availability_doctor(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        return {"availability_doctor": slot_value}

   
    

    def validate_availability_date(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        try:
            appointment_date = datetime.strptime(slot_value, "%Y-%m-%d")
            weekday = appointment_date.weekday()

            if weekday >= 5:
                dispatcher.utter_message(text="Sorry, we are closed on weekends. Please choose a weekday for availability check.")
                return {"appointment_date": None}
            
            # Return valid date - let the form handle the confirmation question
            return {"availability_date": slot_value}
            
        except ValueError:
            dispatcher.utter_message(text="The date format should be YYYY-MM-DD. Please enter a valid date.")
            return {"appointment_date": None}
        

    


class ValidateGetappointmentForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_getappointment_form"
    def validate_full_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        if isinstance(slot_value, str) and len(slot_value.split()) >= 2:
            return {"full_name": slot_value}
        else:
            dispatcher.utter_message(text="Please enter your full name (first and last name).")
            return {"full_name": None}

    
    async def validate_date_of_birth(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        try:
            full_name = tracker.get_slot("full_name")
            
            dob = datetime.strptime(slot_value, "%Y-%m-%d")

            if full_name and dob:
                user_id = get_or_create_user(full_name, slot_value)
                
                if user_id:
                    
                    return {"user_id": user_id}
                else:
                    dispatcher.utter_message("Failed to access user ID.")
                    return {"user_id": None}
        except ValueError:
            dispatcher.utter_message(text="The date format should be YYYY-MM-DD. Please enter a valid date.")
            return {"user_id": None}
        return {"user_id": None}
    
def get_or_create_user(full_name: str, dob: str) -> Optional[int]:
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute(
            'SELECT id FROM "user" WHERE name = %s AND dob = %s',
            (full_name, dob)
        )
        result = cursor.fetchone()

        if result:
            user_id = result[0]
            
        else:
            # Insert new user
            cursor.execute(
                'INSERT INTO "user" (name, dob) VALUES (%s, %s) RETURNING id',
                (full_name, dob)
            )
            user_id = cursor.fetchone()[0]
            
            conn.commit()

        cursor.close()
        conn.close()
        
        return user_id

    except Exception as e:
        print(f"Database error: {e}")
        return None