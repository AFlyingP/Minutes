import { NavLink, Outlet } from "react-router-dom";

const LINKS = [
  { to: "/", id: "nav-search", text: "Search" },
  { to: "/facts", id: "nav-facts", text: "Facts" },
  { to: "/ask", id: "nav-ask", text: "Ask" },
  { to: "/agent", id: "nav-agent", text: "Agent" },
  { to: "/label", id: "nav-label", text: "Label" },
];

export function Placeholder() {
  return <p data-testid="placeholder">Not built yet.</p>;
}

export default function App() {
  return (
    <>
      <header className="header">
        <span className="brand">Minutes</span>
        <nav>
          {LINKS.map((link) => (
            <NavLink key={link.id} to={link.to} data-testid={link.id} end>
              {link.text}
            </NavLink>
          ))}
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
    </>
  );
}
