import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import App, { Placeholder } from "./App";
import SearchPage from "./pages/SearchPage";
import "./styles.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <SearchPage /> },
      { path: "facts", element: <Placeholder /> },
      { path: "ask", element: <Placeholder /> },
      { path: "agent", element: <Placeholder /> },
      { path: "label", element: <Placeholder /> },
      { path: "doc/:id", element: <Placeholder /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
