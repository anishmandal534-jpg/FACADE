from graph_db import graph_db

def seed_initial_persona_facts():
    print("Seeding initial persona facts into Neo4j...")
    
    # Interests and Preferences
    graph_db.add_fact("User", "LIKES", "Quantum Computing")
    graph_db.add_fact("User", "LIKES", "Advanced Mathematics")
    graph_db.add_fact("User", "FAVORITE_MUSIC", "Bollywood")
    graph_db.add_fact("User", "PET", "Persian cat")
    
    # Technical Experience & Projects
    graph_db.add_fact("User", "DEVELOPED", "Web-based flight booking system")
    graph_db.add_fact("User", "WORKS_WITH", "Python")
    graph_db.add_fact("User", "WORKS_WITH", "FastAPI")
    graph_db.add_fact("User", "WORKS_WITH", "Qdrant")
    graph_db.add_fact("User", "WORKS_WITH", "Neo4j")
    
    print("Successfully seeded graph database facts!")

if __name__ == "__main__":
    seed_initial_persona_facts()
    graph_db.close()