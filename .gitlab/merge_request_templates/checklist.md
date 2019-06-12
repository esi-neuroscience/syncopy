Author Guidelines
-----------------
- [ ] Is the change set **< 400 lines**?
- [ ] Was the code checked for memory leaks/performance bottlenecks?
- [ ] Is the code running locally **and** on the ESI cluster?

Reviewer Checklist
------------------
- [ ] Are testing routines present?
- [ ] Do **parallel** loops have a set length and correct termination conditions?
- [ ] Do objects in the global package namespace perform proper parsing of their input? 
- [ ] Can code-blocks be re-factored using builtin/external packages?
- [ ] Code layout
  - [ ] Is the code PEP8 compliant?
  - [ ] Does the code adhere to agreed-upon naming conventions?
  - [ ] Are keywords clearly named and easy to understand?
  - [ ] Is there any commented-out code?
  - [ ] Is a file header present/properly updated?
- [ ] Are all docstrings complete and accurate?