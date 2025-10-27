import psycopg2
from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Tracker,  Action
from rasa_sdk.executor import CollectingDispatcher  
from rasa_sdk.events import SlotSet, FollowupAction, AllSlotsReset
from rasa_sdk.types import DomainDict
import re
DB_PARAMS = {
    'dbname': 'chatbot',
    'user': 'postgres',
    'password': '*********',
    'host': 'localhost',
    'port': '5432'
}


class ActionResetAppointmentSlots(Action):
    def name(self) -> Text:
        return "action_reset_appointment_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Only keep slots if they were set via entity in the latest user message
        slots_to_check = ["appointment_date", "appointment_time", "specific_doctor", "availability_date", "availability_doctor"]
        latest_entities = tracker.latest_message.get("entities", [])
        entity_values = {ent["entity"]: ent["value"] for ent in latest_entities}

        # Keep only slots that were extracted from current message entities
        slots_to_keep = [
            SlotSet(slot, entity_values[slot])
            for slot in slots_to_check if slot in entity_values
        ]

        # Reset all slots, then set back only those just extracted
        return [AllSlotsReset()] + slots_to_keep
    
    
class ActionSaveAppointment(Action):
    def name(self) -> Text:
        return "action_save_appointment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        
        full_name = tracker.get_slot("full_name")
        user_id = tracker.get_slot("user_id")
        date = tracker.get_slot("appointment_date")
        time = tracker.get_slot("appointment_time")
        doctor_name = tracker.get_slot("specific_doctor")
        
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        try:
            if doctor_name and doctor_name.lower() != "no":
            # Check if specific doctor is available at given date and time
                cursor.execute("""
                    SELECT a.id, d.id, d.name FROM appointments a
                    JOIN doctor d ON a.doc_id = d.id
                    WHERE d.name = %s AND a.date = %s AND a.time = %s AND a.is_available = 'yes'
                    LIMIT 1
                """, (doctor_name, date, time))
            else:
                # Randomly get any available doctor at that date & time
                cursor.execute("""
                    SELECT a.id, d.id, d.name FROM appointments a
                    JOIN doctor d ON a.doc_id = d.id
                    WHERE a.date = %s AND a.time = %s AND a.is_available = 'yes'
                    ORDER BY RANDOM()
                    LIMIT 1
                """, (date, time))

            result = cursor.fetchone()

            if not result:
                # Try nearest time for the same doctor on same date
                if doctor_name and doctor_name.lower() != "no":
                    cursor.execute("""
                        SELECT a.time FROM appointments a
                        JOIN doctor d ON a.doc_id = d.id
                        WHERE d.name = %s AND a.date = %s AND a.is_available = 'yes'
                        ORDER BY ABS(EXTRACT(EPOCH FROM (a.time::time - %s::time))) ASC
                        LIMIT 1
                    """, (doctor_name, date, time))
                    nearest_time = cursor.fetchone()

                    if nearest_time:
                        suggested_time = nearest_time[0]
                        dispatcher.utter_message(
                            text=f"Dr. {doctor_name} is not available at {time}, but is available at {suggested_time} on {date}. Would you like to book that slot?"
                        )
                        return [SlotSet("suggested_time", suggested_time), SlotSet("suggested_date", date), SlotSet("suggested_doctor", doctor_name)]
                    else:
                        # Try same time on nearest date
                        cursor.execute("""
                            SELECT a.date FROM appointments a
                            JOIN doctor d ON a.doc_id = d.id
                            WHERE d.name = %s AND a.time = %s AND a.is_available = 'yes'
                            ORDER BY a.date
                            LIMIT 1
                        """, (doctor_name, time))
                        nearest_date = cursor.fetchone()
                        if nearest_date:
                            alt_date = nearest_date[0]
                            dispatcher.utter_message(
                                text=f"Dr. {doctor_name} is not available at {time} on {date}, but is available at the same time on {alt_date}. Would you like to book it?"
                            )
                            return [SlotSet("suggested_time", time), SlotSet("suggested_date", alt_date), SlotSet("suggested_doctor", doctor_name)]
                        else:
                            dispatcher.utter_message(
                                text=f"Sorry, Dr. {doctor_name} is not available near your preferred time. Please try a different time or doctor."
                            )
                            return []
                    
                else:
                    cursor.execute("""
                        SELECT a.time, a.date, d.name FROM appointments a
                        JOIN doctor d ON a.doc_id = d.id
                        WHERE a.date = %s AND a.is_available = 'yes'
                        ORDER BY ABS(EXTRACT(EPOCH FROM (a.time::time - %s::time))) ASC
                        LIMIT 1
                    """, (date, time))
                    nearest_slot = cursor.fetchone()

                    if nearest_slot:
                        nearest_time, nearest_date, suggested_doctor = nearest_slot
                        dispatcher.utter_message(
                            text=f"Unfortunately, no doctors are available at {time} on {date}. "
                                f"But Dr. {suggested_doctor} is available at {nearest_time} on {nearest_date}. "
                                f"Would you like to book that slot?"
                        )
                        return [
                            SlotSet("suggested_time", nearest_time),
                            SlotSet("suggested_date", nearest_date),
                            SlotSet("suggested_doctor", suggested_doctor)
                            
                        ]
                    else:
                        # Try same time on the nearest date for ANY doctor
                        cursor.execute("""
                            SELECT a.date, d.name FROM appointments a
                            JOIN doctor d ON a.doc_id = d.id
                            WHERE a.time = %s AND a.is_available = 'yes'
                            ORDER BY a.date
                            LIMIT 1
                        """, (time,))
                        alt_result = cursor.fetchone()

                        if alt_result:
                            alt_date, suggested_doctor = alt_result
                            dispatcher.utter_message(
                                text=f"No doctors are available at {time} on {date}, but Dr. {suggested_doctor} is available "
                                    f"at {time} on {alt_date}. Would you like to book that slot?"
                            )
                            return [
                                SlotSet("suggested_time", time),
                                SlotSet("suggested_date", alt_date),
                                SlotSet("suggested_doctor", suggested_doctor)
                            ]
                        else:
                            dispatcher.utter_message(
                                text="Sorry, no doctors are available near your requested time. Please try another date or time."
                            )
                            return []
            
            appointment_id, doctor_id, doctor_name_fetched = result

            # Update is_available to 'no' to confirm the booking
            cursor.execute("""
                UPDATE appointments
                SET is_available = 'no', userid = %s
                WHERE id = %s
            """, (user_id, appointment_id,))

            conn.commit()
            

            dispatcher.utter_message(
                f"Appointment booked for {full_name} with Dr. {doctor_name_fetched} on {date} at {time}."
            )

            return [SlotSet("selected_doctor", doctor_name_fetched)]
        finally:
            cursor.close()
            conn.close()



class ActionConfirmSuggestedAppointment(Action):
    def name(self) -> Text:
        return "action_confirm_suggested_appointment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        suggested_date = tracker.get_slot("suggested_date")
        suggested_time = tracker.get_slot("suggested_time")
        suggested_doctor = tracker.get_slot("suggested_doctor")
        # Re-run the appointment booking with updated slots
        return [
            SlotSet("appointment_date", suggested_date),
            SlotSet("appointment_time", suggested_time),
            SlotSet("specific_doctor", suggested_doctor),
            FollowupAction("action_save_appointment")
        ]

class ActiongetAppointment(Action):
    def name(self) -> Text:
        return "action_get_appointment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        user_id = tracker.get_slot("user_id") 
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT a.date, a.time, d.name "
                "FROM appointments a "
                "JOIN doctor d ON a.doc_id = d.id "
                "WHERE a.userid = %s",
                (user_id,)
            )

            result = cursor.fetchone()
            
            if result:
                date, time, name = result
                dispatcher.utter_message(f"You have an appointment with Dr. {name} on {date} at {time}.")
            else:
                dispatcher.utter_message("Sorry, you don't have any booked appointments.")

            return []

        finally:
            conn.commit()
            cursor.close()
            conn.close()

class ActioncancelAppointment(Action):
    def name(self) -> Text:
        return "action_cancel_appointment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        user_id = tracker.get_slot("user_id") 
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT id FROM appointments
                WHERE userid = %s
            """, (user_id,))
            appointment = cursor.fetchone()

            if appointment:
                cursor.execute("""
                    UPDATE appointments
                    SET is_available = 'yes',
                        userid = NULL
                    WHERE userid = %s
                """, (user_id,))
                conn.commit()
                dispatcher.utter_message(text="Your appointment has been successfully cancelled.")
            else:
                dispatcher.utter_message(text="You don't have any appointment to cancel.")

        except Exception as e:
            print(f"Database error: {e}")
            dispatcher.utter_message(text="Sorry, something went wrong while cancelling your appointment.")
        finally:
            
                cursor.close()
                conn.close()

        return []
    

class Actiongetdoctorlist(Action):
    def name(self) -> Text:
        return "action_get_doctor_list"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
       
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM doctor")

            results = cursor.fetchall()
            
            if results:
                doctor_names = ",".join(f" {row[0]}" for row in results)
                dispatcher.utter_message(text=f"Here are the available doctors:{doctor_names}")
            else:
                dispatcher.utter_message(text="No doctors found in the system.")

            return []
        except Exception as e:
            dispatcher.utter_message(text=f"Failed to fetch doctors: {str(e)}")
        finally:
            conn.commit()
            cursor.close()
            conn.close()


class Actiongetdoctoravailability(Action):
    def name(self) -> Text:
        return "action_get_doctor_availability"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        doctor_name = tracker.get_slot("availability_doctor")
        appointment_date = tracker.get_slot("availability_date")
        
        
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM doctor WHERE LOWER(name) = LOWER(%s)", (doctor_name,))
            doctor_row = cursor.fetchone()

            if not doctor_row:
                dispatcher.utter_message(text=f"Doctor '{doctor_name}' not found.")
                return []

            doctor_id = doctor_row[0]

            # Step 2: Fetch available times for that doctor on the given date
            cursor.execute("""
                SELECT time FROM appointments
                WHERE doc_id = %s AND date = %s AND is_available = 'yes'
                ORDER BY time
            """, (doctor_id, appointment_date))
            time_slots = cursor.fetchall()

            if time_slots:
                slot_list = ", ".join(slot[0] for slot in time_slots)
                dispatcher.utter_message(
                    text=f"{doctor_name} is available on {appointment_date} at the following times: {slot_list}"
                )
            else:
                dispatcher.utter_message(
                    text=f"No available appointments for {doctor_name} on {appointment_date}."
                )

            return []
        except Exception as e:
            dispatcher.utter_message(text=f"Error checking availability: {str(e)}")
        finally:
            conn.commit()
            cursor.close()
            conn.close()
