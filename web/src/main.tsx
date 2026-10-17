import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import App, { Placeholder } from "./App";
import DocPage from "./pages/DocPage";
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
      { path: "facts", element: <Placeholder /> },
      { path: "ask", element: <AskPage /> },
      { path: "agent", element: <Placeholder /> },
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
