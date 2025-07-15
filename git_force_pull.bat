@echo off
echo Forcing git pull to get latest changes...

REM Check current status
echo Current git status:
git status

echo.
echo Current branch:
git branch

echo.
echo Remote branches:
git branch -r

echo.
echo Fetching all remote changes...
git fetch --all

echo.
echo Current local commit:
git log --oneline -1

echo.
echo Latest remote commit:
git log --oneline -1 origin/main

echo.
echo Checking if we're behind remote...
git status

echo.
echo Force pulling changes (this will overwrite local changes)...
git reset --hard origin/main

echo.
echo Alternatively, if you want to keep local changes:
echo git pull origin main --rebase

echo.
echo Final status:
git status

pause
