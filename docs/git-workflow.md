# Simple Git/GitHub Workflow for a Two-Person Team

This document outlines a basic workflow to collaborate effectively using Git and GitHub.

## TL;DR - Key Principles

* **Always work on feature branches**: Never commit directly to `main`. Create a new branch for your work (e.g., `git checkout -b feature/my-new-thing`).
* **Update `main` before branching**: Before starting a new feature, ensure your local `main` is up-to-date (`git checkout main; git pull origin main`).
* **Push & Create Pull Request (PR)**: Push your feature branch to GitHub (`git push origin feature/my-new-thing`) and open a PR to merge into `main`.
* **Review PRs**: The other person reviews the code. Discuss and make changes if needed.
* **Merge PR**: Once approved, merge the PR into `main` (usually via the GitHub interface).
* **Sync after merge**: After a PR is merged, everyone should update their local `main` branch (`git checkout main; git pull origin main`).
* **Communicate**: Talk to each other about what you are working on.
* **Clear Commits/PRs**: Write clear commit messages and PR descriptions.

---

## 1. Development Cycle (For Each Feature/Task - Both Persons)

This cycle is repeated for each new feature or significant piece of work.

* **a. Update Local `main` Branch**:
    Before starting new work, always ensure your local `main` branch is up-to-date with the remote `main` branch.

    ```bash
    git checkout main
    git pull origin main
    ```

* **b. Create a Feature Branch**:
    Create a new branch for the specific feature or task. Use a descriptive name.

    ```bash
    # Example: git checkout -b feature/user-authentication
    # Example: git checkout -b bugfix/login-error
    git checkout -b <branch-name>
    ```

* **c. Work on the Feature**:
    * Make your code changes.
    * Commit your changes frequently with clear and concise messages.

    ```bash
    git add .
    git commit -m "feat: Implement user login functionality" # Or a descriptive message
    ```

* **d. Push Feature Branch to GitHub**:
    Once you have made significant progress or are ready for review, push your feature branch to the remote repository.

    ```bash
    git push origin <branch-name>
    ```

* **e. Keeping Your Feature Branch Updated (IMPORTANT!)**:
    If `main` has been updated (e.g., the other person merged their feature) while you are still working on your `<branch-name>`, you should update your feature branch with the latest changes from `main` *before* you create a Pull Request or push final changes. This helps resolve conflicts early.

    **Choose one of these methods:**

    * **Method 1: Rebasing (Preferred for cleaner history)**
        1. Commit or stash any current changes on your `<branch-name>`.
        2. Update your local `main`:

            ```bash
            git checkout main
            git pull origin main
            ```

        3. Switch back to your feature branch and rebase:

            ```bash
            git checkout <branch-name>
            git rebase main
            ```

        4. Resolve any conflicts. Git will guide you. After resolving, `git add .` and `git rebase --continue`.
        5. If you had already pushed `<branch-name>` to GitHub, you'll need to force push (as rebase rewrites history):

            ```bash
            git push origin <branch-name> --force-with-lease
            ```

    * **Method 2: Merging `main` into your feature branch**
        1. Commit or stash any current changes on your `<branch-name>`.
        2. Update your local `main`:

            ```bash
            git checkout main
            git pull origin main
            ```

        3. Switch back to your feature branch and merge `main` into it:

            ```bash
            git checkout <branch-name>
            git merge main
            ```

        4. Resolve any conflicts and commit the merge.
        5. Push your branch:

            ```bash
            git push origin <branch-name>
            ```

## 2. Code Review and Merge via Pull Request (PR)

* **a. Create a Pull Request (PR)**:
    * Go to the GitHub repository page.
    * You should see a prompt to create a Pull Request for your recently pushed branch. Click it.
    * Alternatively, go to the "Pull requests" tab and click "New pull request".
    * Select your feature branch (e.g., `<branch-name>`) to be merged into the `main` branch.
    * Add a clear title and description for your PR, explaining the changes.
    * Assign the other person as a reviewer.

* **b. Review the PR (The other person)**:
    * The reviewer receives a notification.
    * Reviews the code changes, adds comments, and asks questions if needed.
    * If changes are requested, the person who created the PR makes the necessary updates on their feature branch, commits them, and pushes again. The PR will update automatically.

* **c. Merge the PR**:
    * Once the review is complete, approved, and any CI checks pass, the PR can be merged into the `main` branch.
    * This is typically done using the "Merge pull request" button on GitHub.
    * **Important**: Delete the feature branch on GitHub after merging to keep the repository clean. GitHub usually offers a button for this.

## 3. Keep Local Repositories Updated

* After a PR is merged into `main` on GitHub, both persons should update their local `main` branch:

    ```bash
    git checkout main
    git pull origin main
    ```

* You can also delete your local feature branch if you are done with it:

    ```bash
    git branch -d <branch-name>
    ```

    If Git complains that the branch is not fully merged (this can happen if the merge strategy on GitHub was "squash" or "rebase"), you can use `git branch -D <branch-name>` to force delete it, assuming you've already pulled the changes into `main`.

---

This workflow keeps the `main` branch stable and allows for parallel work and code review.
