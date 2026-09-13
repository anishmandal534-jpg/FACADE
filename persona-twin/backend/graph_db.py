import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()


class GraphDatabaseManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")

        self.driver = None

        if not self.uri or not self.password:
            print(
                "NEO4J_URI / NEO4J_PASSWORD not set. "
                "Neo4j graph operations will be disabled."
            )
            return

        self.connect()

    def connect(self):
        """Connect to Neo4j Aura."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )

            self.driver.verify_connectivity()

            print("Graph Manager successfully connected to Neo4j Aura!")

        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            self.driver = None

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    # ============================================================
    # ADD FACT
    # ============================================================

    def add_fact(self, entity, relation, target):
        """Add a fact/relation to the graph."""

        if not self.driver:
            return False

        try:
            with self.driver.session() as session:

                # Relationship types cannot be parameterized in Cypher.
                # Sanitize the relation before inserting it.
                safe_relation = "".join(
                    c for c in str(relation)
                    if c.isalnum() or c == "_"
                ).upper()

                if not safe_relation:
                    safe_relation = "RELATED_TO"

                query = f"""
                MERGE (e:Entity {{name: $entity}})
                MERGE (t:Entity {{name: $target}})
                MERGE (e)-[r:{safe_relation}]->(t)
                """

                session.run(
                    query,
                    entity=str(entity).strip(),
                    target=str(target).strip()
                )

            return True

        except Exception as e:
            print(f"Neo4j add_fact error: {e}")
            return False

    # ============================================================
    # GET RELATED FACTS
    # ============================================================

    def get_related_facts(self, entity_name):
        """
        Get facts/relationships connected to a persona.

        Example:
            Affan STUDIES Computer Science
            Affan LIKES Minecraft
        """

        if not self.driver:
            return []

        facts = []

        try:
            with self.driver.session() as session:

                query = """
                MATCH (e:Entity {name: $entity_name})-[r]->(t)
                RETURN
                    e.name AS subject,
                    type(r) AS relation,
                    t.name AS target
                LIMIT 10
                """

                result = session.run(
                    query,
                    entity_name=str(entity_name).strip()
                )

                for record in result:

                    subject = record["subject"]
                    relation = record["relation"]
                    target = record["target"]

                    if subject and relation and target:
                        facts.append(
                            f"{subject} {relation} {target}"
                        )

        except Exception as e:
            print(f"Neo4j get_related_facts error: {e}")

        return facts

    # ============================================================
    # ADD TRAIT
    # ============================================================

    def add_trait(self, entity, trait):
        """Add a trait to a persona/entity."""

        if not self.driver:
            return False

        try:
            with self.driver.session() as session:

                query = """
                MERGE (e:Entity {name: $entity})
                MERGE (t:Trait {name: $trait})
                MERGE (e)-[:HAS_TRAIT]->(t)
                """

                session.run(
                    query,
                    entity=str(entity).strip(),
                    trait=str(trait).strip()
                )

            return True

        except Exception as e:
            print(f"Neo4j add_trait error: {e}")
            return False

    # ============================================================
    # GET PERSONA TRAITS
    # ============================================================

    def get_persona_traits(self, entity_name):
        """Return all traits associated with a persona."""

        if not self.driver:
            return []

        try:
            with self.driver.session() as session:

                query = """
                MATCH (e:Entity {name: $entity_name})
                      -[:HAS_TRAIT]->
                      (t:Trait)
                RETURN t.name AS trait
                ORDER BY t.name
                """

                result = session.run(
                    query,
                    entity_name=str(entity_name).strip()
                )

                return [
                    record["trait"]
                    for record in result
                    if record["trait"]
                ]

        except Exception as e:
            print(f"Neo4j get_persona_traits error: {e}")
            return []

    # ============================================================
    # DELETE ONE TRAIT
    # ============================================================

    def delete_trait(self, entity, trait):
        """Delete one trait from a persona."""

        if not self.driver:
            return False

        try:
            with self.driver.session() as session:

                query = """
                MATCH (e:Entity {name: $entity})
                      -[r:HAS_TRAIT]->
                      (t:Trait)
                WHERE toLower(t.name) = toLower($trait)
                DELETE r
                """

                session.run(
                    query,
                    entity=str(entity).strip(),
                    trait=str(trait).strip()
                )

            return True

        except Exception as e:
            print(f"Neo4j delete_trait error: {e}")
            return False

    # ============================================================
    # DELETE ALL PERSONA TRAITS
    # ============================================================

    def delete_persona_traits(self, entity_name):
        """Delete all traits belonging to a persona."""

        if not self.driver:
            return False

        try:
            with self.driver.session() as session:

                query = """
                MATCH (e:Entity {name: $entity_name})
                      -[r:HAS_TRAIT]->
                      (t:Trait)
                DELETE r
                """

                session.run(
                    query,
                    entity_name=str(entity_name).strip()
                )

            return True

        except Exception as e:
            print(f"Neo4j delete_persona_traits error: {e}")
            return False


# ================================================================
# GLOBAL NEO4J MANAGER
# ================================================================

graph_db = GraphDatabaseManager()