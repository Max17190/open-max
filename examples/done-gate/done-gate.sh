# Run by done-gate.toml at each end of a turn, with the check as its argument.
# Exit 0 lets the end stand. Exit 1 refuses it, and what this prints (Open Max
# keeps the first 500 characters) goes back to the model as a user message.
check=${1:?usage: sh done-gate.sh '<check command>'}
# The turn_end payload on stdin is not needed. Always run the check: an end
# that cannot be refused (blockable false) still needs an honest exit, since
# exit 0 there would record the end as verified.
cat > /dev/null
output=$(sh -c "$check" 2>&1) && exit 0
status=$?
printf 'Not done: `%s` exits %s. Fix the cause, then end the turn again.\n' "$check" "$status"
printf '%s\n' "$output" | tail -c 380
exit 1
