DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'minutes_ro') THEN
    CREATE ROLE minutes_ro LOGIN PASSWORD 'minutes_ro'
      NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION;
  END IF;
END
$$;
ALTER ROLE minutes_ro SET default_transaction_read_only = on;
ALTER ROLE minutes_ro SET statement_timeout = '3000ms';
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM minutes_ro;
REVOKE ALL ON SCHEMA public FROM minutes_ro;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO minutes_ro;
GRANT SELECT ON v_meetings, v_documents, v_units, v_items, v_facts, v_motions, v_votes,
  v_vote_members, v_ordinances, v_amounts, v_statements TO minutes_ro;
