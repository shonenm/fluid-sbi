Commit the changes.

Follow these steps:

1. Run `git status` to check changes
2. Run `git diff` to review staged/unstaged changes
3. Run `git log --oneline -5` to check recent commit style
4. **Split changes into logical units**:
   - Separate commits for different purposes
   - Example: feature addition and bug fix = separate commits
   - Use `git add <file>` to stage only related files
5. Create appropriate commit message:
   - Use English
   - Include type prefix: feat, fix, docs, refactor, chore, test
   - Keep summary concise (1 line)
6. Stage and commit (repeat 4-6 if splitting)

Commit message format:
```
<type>(<scope>): <summary>

<optional body>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

Examples:
- `feat(config): add Pydantic configuration management`
- `fix(score): resolve type errors`
- `docs(general): add development setup guide`
- `refactor(utils): organize import statements`
- `chore(deps): update dependencies`
- `test(config): add unit tests for config module`
