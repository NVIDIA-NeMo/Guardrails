// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(function () {
    function findNextTable(button) {
        let current = button.nextElementSibling;

        while (current) {
            if (current.tagName && current.tagName.toLowerCase() === "table") {
                return current;
            }

            const nestedTable = current.querySelector ? current.querySelector("table") : null;
            if (nestedTable) {
                return nestedTable;
            }

            current = current.nextElementSibling;
        }

        return null;
    }

    function createModal() {
        const modal = document.createElement("div");
        modal.className = "table-expander-modal";
        modal.setAttribute("role", "dialog");
        modal.setAttribute("aria-modal", "true");
        modal.setAttribute("aria-hidden", "true");

        const dialog = document.createElement("div");
        dialog.className = "table-expander-modal__dialog";

        const header = document.createElement("div");
        header.className = "table-expander-modal__header";

        const title = document.createElement("p");
        title.className = "table-expander-modal__title";

        const closeButton = document.createElement("button");
        closeButton.className = "table-expander-modal__close";
        closeButton.type = "button";
        closeButton.setAttribute("aria-label", "Close expanded table");
        closeButton.textContent = "Close";

        const body = document.createElement("div");
        body.className = "table-expander-modal__body";

        header.append(title, closeButton);
        dialog.append(header, body);
        modal.append(dialog);
        document.body.append(modal);

        return { body, closeButton, modal, title };
    }

    function initializeTableExpanders() {
        const buttons = document.querySelectorAll(".table-expand-button");
        if (!buttons.length) {
            return;
        }

        const modalParts = createModal();
        let activeButton = null;

        function closeModal() {
            modalParts.modal.classList.remove("is-open");
            modalParts.modal.setAttribute("aria-hidden", "true");
            document.body.classList.remove("table-expander-modal-open");
            modalParts.body.replaceChildren();

            if (activeButton) {
                activeButton.focus();
                activeButton = null;
            }
        }

        function openModal(button, table) {
            const clonedTable = table.cloneNode(true);
            activeButton = button;
            modalParts.title.textContent = button.dataset.tableTitle || "Expanded table";
            modalParts.body.replaceChildren(clonedTable);
            modalParts.modal.classList.add("is-open");
            modalParts.modal.setAttribute("aria-hidden", "false");
            document.body.classList.add("table-expander-modal-open");
            modalParts.closeButton.focus();
        }

        buttons.forEach((button) => {
            const table = findNextTable(button);

            if (!table) {
                button.hidden = true;
                return;
            }

            button.setAttribute("aria-label", "Open table in expanded view");
            button.addEventListener("click", () => openModal(button, table));
        });

        modalParts.closeButton.addEventListener("click", closeModal);
        modalParts.modal.addEventListener("click", (event) => {
            if (event.target === modalParts.modal) {
                closeModal();
            }
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && modalParts.modal.classList.contains("is-open")) {
                closeModal();
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeTableExpanders);
    } else {
        initializeTableExpanders();
    }
})();
