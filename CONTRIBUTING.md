## Contributing

0. Assume that you have clone the git. Otherwise
```
git clone https://github.com/Linsan5768/capstone_Lin.git
```
1.	Git pull from the main remote branch to your local main branch
```
git checkout main
git pull
```
(Or alternatively, instead of ```git pull``` you can use ```git fetch origin``` then ```git merge origin/main``` in case if you have merge conflict)\
2.	To implement new feature or make change, create your own local test branch
```
git branch <your-feature-branch-name>
git checkout <your-feature-branch-name>
```
3. After works done, don't forget to commit your changes
```
git add <file-that-you-want-to-add>
git commit -m <meaningful-commit-message>
```
4. Push your changes to remote branch
```
git push -u origin <your-feature-branch-name>
```
5. You should be able to see your changes on GitHub with your branch that you just created. After review, you can merge your branch to main.

For any issues or improvements, please contact [Linsan5768](https://github.com/Linsan5768).