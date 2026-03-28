# Columns Introduction - Available Ingredients

## 1. `data/user_subset.json` (Active User Subset - Yelp Only)

This file compiles the characteristics of the most active Yelp users (redundant `source` and `type` fields removed).

### Core Fields
- **user_id**: Unique identifier for the user (Primary Key). Used to map across tables.
- **name**: User’s display name or nickname.
- **yelping_since**: Timestamp when the user joined Yelp (`YYYY-MM-DD HH:MM:SS`).
- **review_count**: Total number of reviews published.
- **average_stars**: Average rating given by the user (`1.0` to `5.0`).
- **fans**: Number of followers (indicator of social influence).
- **elite**: Years the user earned "Yelp Elite" status (string or array).
- **friends**: List of connected `user_id`s.

### User Feedback Metrics
- **useful**: Total "Useful" votes received.
- **funny**: Total "Funny" votes received.
- **cool**: Total "Cool" votes received.

### Compliments (Badge System)
- **compliment_hot**
- **compliment_more**
- **compliment_profile**
- **compliment_cute**
- **compliment_list**
- **compliment_note**
- **compliment_plain**
- **compliment_cool**
- **compliment_funny**
- **compliment_writer**
- **compliment_photos**

These represent different types of appreciation from other users.

---

## 2. `data/item_subset.json` (Entity Item Subset - Yelp Only)

This file contains business/entity data reviewed by active users.

### Core Fields
- **item_id**: Unique identifier for the business (Primary Key).
- **name**: Business name.
- **stars**: Average rating (`1.0` to `5.0`).
- **review_count**: Total number of reviews.
- **is_open**: Operational status (`1` = open, `0` = permanently closed).

### Geographic Information
- **address**: Street address.
- **city**: City location.
- **state**: State/province abbreviation.
- **postal_code**: ZIP/postal code.
- **latitude**: Geographic latitude.
- **longitude**: Geographic longitude.

### Additional Features
- **categories**: Business categories (e.g., `"Restaurants, Italian, Coffee & Tea"`).
- **hours**: Operating hours per day (e.g., `"Monday": "8:0-22:0"`).
- **attributes**: Dictionary of business features:
  - **BusinessParking**: Parking availability
  - **WiFi**: Internet access
  - **GoodForKids**: Child-friendly indicator
  - **RestaurantsPriceRange2**: Price range

These fields are useful for feature engineering, NLP, and recommendation systems.

---

## 3. `data/review_subset.json` (Interaction / Review Subset - Yelp Only)

This file represents the relationship between users and businesses.

### Entity Relationship Keys (Foreign Keys)
- **review_id**: Unique identifier for each review.
- **user_id**: References `user_subset.json`.
- **item_id**: References `item_subset.json`.

### Review Content (Content & Sentiment)
- **stars**: Rating given (`1.0` to `5.0`)  
  → Often used as the **target label** in prediction models.
- **text**: Review text (unstructured data for NLP tasks).
- **date**: Timestamp of review (`YYYY-MM-DD HH:MM:SS`)  
  → Used for temporal analysis and dataset splitting.

### Social Feedback (Review-Level Metrics)
- **useful**: Number of "Useful" votes.
- **funny**: Number of "Funny" votes.
- **cool**: Number of "Cool" votes.

> **Note:** These metrics are often used as **review quality weights**.  
> Reviews with higher "useful" counts tend to be more reliable and informative.

---

## Summary

- **user_subset.json** → User features (who)
- **item_subset.json** → Business features (what)
- **review_subset.json** → Interactions (who ↔ what)

Together, these form a complete dataset for:
- Recommendation systems
- Sentiment analysis
- User behavior modeling
- Graph-based learning
- LLM-powered agents