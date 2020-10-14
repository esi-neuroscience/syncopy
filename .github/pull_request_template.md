Author Guidelines
-----------------
- [ ] Is the change set **< 400 lines**?
- [ ] Was the code checked for memory leaks/performance bottlenecks?
- [ ] Is the code running locally **and** on the ESI cluster?
- [ ] Is the code running on all supported platforms?

Reviewer Checklist
------------------
- [ ] Are testing routines present?
- [ ] Do **parallel** loops have a set length and correct termination conditions?
- [ ] Do objects in the global package namespace perform proper parsing of their input? 
- [ ] Do code-blocks provide novel functionality, i.e., no re-factoring using builtin/external packages possible?
- [ ] Code layout
  - [ ] Is the code PEP8 compliant?
  - [ ] Does the code adhere to agreed-upon naming conventions?
  - [ ] Are keywords clearly named and easy to understand?
  - [ ] No commented-out code?
- [ ] Are all docstrings complete and accurate?
