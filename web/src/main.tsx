import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import App from "./App";
import AgentPage from "./pages/AgentPage";
import DocPage from "./pages/DocPage";
import FactsPage from "./pages/FactsPage";
import LabelPage from "./pages/LabelPage";
import AskPage from "./pages/AskPage";
import SearchPage from "./pages/SearchPage";
import "./styles.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <SearchPage /> },
      { path: "facts", element: <FactsPage /> },
      { path: "ask", element: <AskPage /> },
      { path: "agent", element: <AgentPage /> },
      { path: "label", element: <LabelPage /> },
      { path: "doc/:id", element: <DocPage /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
