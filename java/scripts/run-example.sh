#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: npm run run <example_folder> [MAIN=ClassName]"
  exit 1
fi

EXAMPLE=$1
MAIN_CLASS=${MAIN:-$(ls examples/src/main/java/examples/${EXAMPLE}/*.java 2>/dev/null | head -n1 | xargs -n1 basename 2>/dev/null | sed 's/.java$//')}

if [ -z "$MAIN_CLASS" ]; then
  echo "Error: Could not find example in examples/src/main/java/examples/${EXAMPLE}/"
  exit 1
fi

mvn -q -f examples/pom.xml -DskipTests -Dexec.cleanupDaemonThreads=false -Dexec.mainClass=examples.${EXAMPLE}.${MAIN_CLASS} clean compile exec:java

