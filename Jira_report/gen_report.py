"""
JIRA Report Generator

This script connects to a JIRA instance to extract worklog data for a specified project within a given date range. 
It processes the worklogs to calculate time spent by each person on various tasks and epics, and generates 
HTML reports, CSV files, and JSON outputs. The script also includes functionality to send the generated report 
via email.

Key Features:
- Connects to JIRA using provided credentials.
- Extracts worklogs based on specified project and date range.
- Converts worklogs to JSON format for easy manipulation.
- Calculates time spent per person and per epic.
- Generates visual reports in HTML format with pie charts.
- Sends the report via email using SMTP.

Dependencies:
- JIRA Python library
- Pandas
- Matplotlib
- BeautifulSoup
- dotenv
"""

# Import necessary libraries
import base64
import io
import json
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from jira import JIRA

# Load environment variables from a .env file
load_dotenv()

# Configure logging to display information and error messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_jira_connection(server, email, token):
    """Establish a connection to the JIRA server."""
    return JIRA(server=server, basic_auth=(email, token))

def send_html_email(subject, sender_email, receiver_email, smtp_server, smtp_port, smtp_username, smtp_password, html_content):
    """Send an HTML email using SMTP."""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    # Attach the HTML content to the email
    part2 = MIMEText(html_content, 'html')
    msg.attach(part2)

    try:
        # Connect to the SMTP server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            logging.info("Email sent successfully")
    except Exception as e:
        logging.error(f"Error sending email: {e}")

def convert_worklogs_to_json(df):
    """Convert worklog DataFrame to a JSON structure."""
    json_data = {}
    grouped_epics = df.groupby(['Epic Key', 'Epic Name', 'Epic Labels'])

    for (epic_key, epic_name, epic_labels), epic_group in grouped_epics:
        if epic_key not in json_data:
            json_data[epic_key] = {
                'Epic Name': epic_name,
                'Epic Labels': [label.strip() for label in epic_labels.split(',')] if isinstance(epic_labels, str) else [],
                'Issues': {}
            }

        grouped_issues = epic_group.groupby(['Issue Key', 'Summary', 'Issue Type'])
        for (issue_key, summary, issue_type), issue_group in grouped_issues:
            if issue_key not in json_data[epic_key]['Issues']:
                json_data[epic_key]['Issues'][issue_key] = {
                    'Summary': summary,
                    'Issue Type': issue_type,
                    'Worklogs': []
                }

            for _, row in issue_group.iterrows():
                worklog = {
                    'Worklog Author': row['Worklog Author'],
                    'Time Spent': row['Time Spent'],
                    'Worklog Created': row['Worklog Created'],
                    'Worklog Comment': row['Worklog Comment']
                }
                json_data[epic_key]['Issues'][issue_key]['Worklogs'].append(worklog)

    return json.dumps(json_data, indent=4)

def time_to_hours(time_str):
    """Convert JIRA time string to hours."""
    if time_str.endswith('d'):
        return float(time_str[:-1]) * 8  # Convert days to hours
    elif time_str.endswith('h'):
        return float(time_str[:-1])  # Hours remain the same
    elif time_str.endswith('m'):
        return float(time_str[:-1]) / 60  # Convert minutes to hours
    elif time_str.endswith('s'):
        return float(time_str[:-1]) / 3600  # Convert seconds to hours
    return 0

def calculate_time_per_person_and_label(worklogs_df):
    """Calculate time spent by each person and label."""
    worklogs_df['Hours Spent'] = worklogs_df['Time Spent'].apply(time_to_hours)
    worklogs_df['Epic Label'] = worklogs_df['Epic Labels'].str.split(',')
    worklogs_df = worklogs_df.explode('Epic Label').reset_index(drop=True)
    worklogs_df['Epic Label'] = worklogs_df['Epic Label'].str.strip()

    time_per_person_and_label = worklogs_df.groupby(['Worklog Author', 'Epic Label'])['Hours Spent'].sum().reset_index()
    time_per_person_and_label['Hours Spent'] = time_per_person_and_label['Hours Spent'].round(2)

    return time_per_person_and_label

def filter_time_by_label(total_time_per_person_and_label, labels_to_exclude):
    """Filter time data by excluding specified labels."""
    mask_exclude = ~total_time_per_person_and_label['Epic Label'].isin(labels_to_exclude)
    mask_include = total_time_per_person_and_label['Epic Label'].isin(labels_to_exclude)
    
    filtered_df_exclude = total_time_per_person_and_label[mask_exclude].copy()
    filtered_df_include = total_time_per_person_and_label[mask_include].copy()
    
    filtered_df_exclude.reset_index(drop=True, inplace=True)
    filtered_df_include.reset_index(drop=True, inplace=True)
    
    return filtered_df_exclude, filtered_df_include

def extract_worklogs_in_period(jira, project, start_date, end_date, epic_field_id):
    """
    Extract worklogs entered in a JIRA project within a given period.

    :param jira: Connected JIRA instance
    :param project: JIRA project key (e.g., 'PROJ')
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :param epic_field_id: Custom field ID for Epic Link
    :return: DataFrame containing the extracted worklog information
    """

    # Convert dates to datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # JQL to search for issues in the project that have worklogs
    jql = f'''project = {project} AND (
        worklogDate >= "{start_date}" AND worklogDate <= "{end_date}"
        OR (
            status changed to Done 
            DURING ("{start_date}", "{end_date}")
        )
    )'''

    # Fields to retrieve
    fields = 'key,summary,issuetype,{epic_field_id},labels,parent'.format(epic_field_id=epic_field_id)

    try:
        # Search for issues
        issues = jira.search_issues(jql, fields=fields, expand='worklog', maxResults=False)
        logging.info(f"Retrieved {len(issues)} issues from JIRA.")
    except Exception as e:
        logging.error(f"Error searching issues: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

    # List to store data
    data = []
    epic_cache = {}

    for issue in issues:
        issue_type = issue.fields.issuetype.name
        epic_key = None
        epic_name = ''
        epic_labels = []
        summary = issue.fields.summary

        # If it's an Epic
        if issue_type == 'Epic':
            epic_key = issue.key
            epic_name = issue.fields.summary
            epic_labels = issue.fields.labels if hasattr(issue.fields, 'labels') else []
        
        # If it's a sub-task
        elif getattr(issue.fields.issuetype, 'subtask', False):
            try:
                # Check if the parent task exists
                parent = getattr(issue.fields, 'parent', None)
                if parent is None:
                    logging.warning(f"Subtask {issue.key} has no parent task")
                    continue

                parent_key = parent.key
                parent_issue = jira.issue(parent_key)
                
                # Combine the parent task name with the sub-task
                summary = f"{parent_issue.fields.summary} --> {issue.fields.summary}"
                
                # Retrieve the Epic of the parent task
                parent_epic_key = getattr(parent_issue.fields, epic_field_id, None)
                
                if parent_epic_key:
                    if parent_epic_key in epic_cache:
                        epic_details = epic_cache[parent_epic_key]
                        epic_key = parent_epic_key
                        epic_name = epic_details['Epic Name']
                        epic_labels = epic_details['Epic Labels']
                    else:
                        epic_issue = jira.issue(parent_epic_key)
                        epic_key = parent_epic_key
                        epic_name = epic_issue.fields.summary
                        epic_labels = epic_issue.fields.labels
                        epic_cache[parent_epic_key] = {
                            'Epic Name': epic_name,
                            'Epic Labels': epic_labels
                        }
                else:
                    logging.warning(f"Parent task {parent_key} of subtask {issue.key} has no epic")
            except Exception as e:
                logging.error(f"Error retrieving epic details for subtask {issue.key}: {str(e)}")
        
        # For all other task types
        else:
            epic_key = getattr(issue.fields, epic_field_id, None)
            if epic_key:
                if epic_key in epic_cache:
                    epic_details = epic_cache[epic_key]
                    epic_name = epic_details['Epic Name']
                    epic_labels = epic_details['Epic Labels']
                else:
                    try:
                        epic_issue = jira.issue(epic_key)
                        epic_name = epic_issue.fields.summary
                        epic_labels = epic_issue.fields.labels
                        epic_cache[epic_key] = {
                            'Epic Name': epic_name,
                            'Epic Labels': epic_labels
                        }
                    except Exception as e:
                        logging.error(f"Error retrieving epic {epic_key}: {e}")
                        epic_cache[epic_key] = {'Epic Name': '', 'Epic Labels': []}

        # Retrieve work logs
        try:
            worklogs = jira.worklogs(issue.key)
        except Exception as e:
            logging.error(f"Error retrieving worklogs for issue {issue.key}: {e}")
            continue  # Skip to the next issue in case of error

        # Check if the task was completed during the period
        status_changes = jira.issue(issue.key, expand='changelog').changelog.histories
        done_during_period = False
        done_by = None
        
        for history in status_changes:
            change_date = datetime.strptime(history.created[:10], '%Y-%m-%d')
            if start_dt <= change_date <= end_dt:
                for item in history.items:
                    if item.field == 'status' and item.toString == 'Done':
                        done_during_period = True
                        # Retrieve the author of the change
                        author_name = history.author.displayName
                        done_by = ''.join(word[0].upper() for word in author_name.split())
                        break
                if done_during_period:
                    break

        # If the task has no worklogs but was completed during the period
        if done_during_period and not worklogs:
            data.append({
                'Issue Key': issue.key,
                'Summary': summary,
                'Issue Type': issue.fields.issuetype.name,
                'Epic Key': epic_key,
                'Epic Name': epic_name,
                'Epic Labels': ', '.join(epic_labels) if epic_labels else '',
                'Worklog Author': done_by,  # Use the author of the change
                'Time Spent': '0h',
                'Worklog Created': '',
                'Worklog Comment': '[Completed during this period]'
            })

        # Continue with existing worklog processing
        for worklog in worklogs:
            try:
                # Convert worklog.created to datetime
                worklog_created = datetime.strptime(worklog.created[:10], '%Y-%m-%d')
                if start_dt <= worklog_created <= end_dt:
                    # Get worklog comment if exists
                    worklog_comment = getattr(worklog, 'comment', '') or ''

                    data.append({
                        'Issue Key': issue.key,
                        'Summary': summary,
                        'Issue Type': issue.fields.issuetype.name,
                        'Epic Key': epic_key,
                        'Epic Name': epic_name,
                        'Epic Labels': ', '.join(epic_labels) if epic_labels else '',
                        'Worklog Author': worklog.author.displayName,
                        'Time Spent': worklog.timeSpent,
                        'Worklog Created': worklog.created,
                        'Worklog Comment': worklog_comment
                    })
            except AttributeError as ae:
                logging.error(f"Attribute error processing worklog for issue {issue.key}: {ae}")
            except Exception as e:
                logging.error(f"Error processing worklog for issue {issue.key}: {e}")

    # Create a DataFrame
    df = pd.DataFrame(data)

    logging.info(f"Extracted {len(df)} worklogs from JIRA.")
    return df

def calculate_time_per_person_and_epic(worklogs_df):
    """
    Calculate the time spent by each person on each Epic.

    :param worklogs_df: DataFrame containing worklog data
    :return: DataFrame with time spent per person and per Epic
    """
    # Convert 'Time Spent' to hours
    def time_to_hours(time_str):
        if time_str.endswith('d'):
            return float(time_str[:-1]) * 8  # 1 day = 8 hours
        elif time_str.endswith('h'):
            return float(time_str[:-1])
        elif time_str.endswith('m'):
            return float(time_str[:-1]) / 60
        elif time_str.endswith('s'):
            return float(time_str[:-1]) / 3600
        else:
            return 0

    worklogs_df['Hours Spent'] = worklogs_df['Time Spent'].apply(time_to_hours)

    # Group by author and Epic, then sum the time spent
    time_per_person_and_epic = worklogs_df.groupby(['Worklog Author', 'Epic Name'])['Hours Spent'].sum().reset_index()

    # Round to two decimal places
    time_per_person_and_epic['Hours Spent'] = time_per_person_and_epic['Hours Spent'].round(2)

    return time_per_person_and_epic

def create_graph_from_filtered_df(filtered_df_exclude):
    """
    Create a pie chart from the filtered DataFrame.

    :param filtered_df_exclude: DataFrame containing filtered data
    :return: None (displays the graph)
    """
    import matplotlib.pyplot as plt

    # Calculate total hours per Epic label
    total_hours_per_label = filtered_df_exclude.groupby('Epic Label')['Hours Spent'].sum()

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a pie chart
    ax.pie(total_hours_per_label.values, labels=total_hours_per_label.index, autopct='%1.1f%%', startangle=90)

    # Customize the graph
    plt.title("Time Distribution by Epic Label")
    plt.axis('equal')  # Ensures the pie chart is circular

    # Display the graph
    plt.show()

    # Optionally, save the graph
    # plt.savefig('graph_filtered_df_pie.png')

def create_csv_from_json(worklogs_json):
    """
    Create a CSV string from JSON worklog data.

    :param worklogs_json: JSON string containing worklog data
    :return: CSV string containing worklog details
    """
    import json
    import csv
    from io import StringIO

    data = json.loads(worklogs_json)
    csv_output = StringIO()
    csv_writer = csv.writer(csv_output)

    csv_writer.writerow(['Epic', 'Task Key', 'Task', 'Task Type', 'Worklog Author', 'Comment'])

    previous_row = None
    for epic_key, epic_data in data.items():
        epic_name = epic_data['Epic Name']
        for issue_key, issue_data in epic_data['Issues'].items():
            issue_summary = issue_data['Summary']
            issue_type = issue_data['Issue Type']
            for worklog in issue_data['Worklogs']:
                current_row = [
                    epic_name,
                    issue_key,
                    issue_summary,
                    issue_type,
                    worklog['Worklog Author'],
                    worklog['Worklog Comment']
                ]
                if current_row != previous_row:
                    if current_row[5] or (previous_row is None or current_row[:5] != previous_row[:5]):
                        csv_writer.writerow(current_row)
                        previous_row = current_row

    return csv_output.getvalue()

def create_html_table_from_json(worklogs_json, filtered_df_exclude, filtered_df_include, planned_tasks_df):
    """
    Create an HTML table from worklog JSON data and planned tasks.
    """
    import json
    import matplotlib.pyplot as plt
    import io
    import base64

    # Function to create a pie chart and convert it to base64
    def create_pie_chart(df, title, type_colour):
        plt.figure(figsize=(8, 6))
        total_hours_per_label = df.groupby('Epic Label')['Hours Spent'].sum()
        
        # Color palette for non-standard labels (left chart)
        other_colors = [
            '#FF9999',  # Light pink
            '#66B2FF',  # Light blue
            '#99FF99',  # Light green
            '#FFCC99',  # Light orange
            '#FF99CC',  # Dark pink
            '#99CCFF',  # Pastel blue
            '#FFB366',  # Dark orange
            '#99FF99',  # Pastel green
            '#FF99FF',  # Light violet
            '#99FFCC',  # Turquoise
        ]
        
        # Define colors corresponding to standard labels
        standard_color_mapping = {
            'B2C': '#3498db',      # Blue
            'B2B': '#e74c3c',      # Red
            'Licensing': '#2ecc71', # Green
            'RI': '#f1c40f',       # Yellow
            'No Label': '#95a5a6'  # Gray
        }
        
        # Create the list of colors based on the chart title
        if type_colour == 1:  # Right chart
            colors = [standard_color_mapping.get(label, '#95a5a6') for label in total_hours_per_label.index]
        else:  # Left chart
            # Sort labels alphabetically and assign colors
            sorted_labels = sorted(total_hours_per_label.index)
            color_dict = dict(zip(sorted_labels, other_colors[:len(sorted_labels)]))
            colors = [color_dict[label] for label in total_hours_per_label.index]
        
        plt.pie(total_hours_per_label.values, 
                labels=total_hours_per_label.index, 
                autopct='%1.1f%%', 
                colors=colors)
        
        plt.title(title)
        plt.axis('equal')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode()

    # Create the two charts
    chart1 = create_pie_chart(filtered_df_exclude, "Time Distribution by Project", 0)
    chart2 = create_pie_chart(filtered_df_include, "Time Distribution", 1)

    data = json.loads(worklogs_json)
    
    # Create a unified structure for worklogs and planned tasks
    unified_data = {
        'B2C': {},
        'B2B': {},
        'Licensing': {},
        'RI': {},
        'No Label': {}
    }

    # First, create a dictionary of epics from worklogs_json to have the correct labels
    epic_label_mapping = {}  # To store {epic_key: label}
    
    # Add worklogs to the unified structure and build the epic mapping
    for epic_key, epic_data in data.items():
        epic_labels = epic_data.get('Epic Labels', [])
        assigned_label = 'No Label'
        for label in ['B2C', 'B2B', 'Licensing', 'RI']:
            if label in epic_labels:
                assigned_label = label
                break
        
        epic_label_mapping[epic_key] = assigned_label
        
        if epic_key not in unified_data[assigned_label]:
            unified_data[assigned_label][epic_key] = {
                'epic_name': epic_data['Epic Name'],
                'worklogs': epic_data['Issues'],
                'planned_tasks': []
            }

    # Add planned tasks using the existing epic mapping
    for _, task in planned_tasks_df.iterrows():
        epic_info = task['Epic'].split(' - ', 1) if ' - ' in task['Epic'] else ['No Epic', task['Epic']]
        epic_key = epic_info[0]
        
        # If the epic already exists in our mapping, use its label
        if epic_key in epic_label_mapping:
            assigned_label = epic_label_mapping[epic_key]
        else:
            # Otherwise, determine the label from the epic name or summary
            epic_name = epic_info[1]
            assigned_label = 'No Label'
            for label in ['B2C', 'B2B', 'Licensing', 'RI']:
                if label.lower() in epic_name.lower() or label.lower() in task['Summary'].lower():
                    assigned_label = label
                    break
        
        # If the epic does not yet exist in the structure, add it
        if epic_key not in unified_data[assigned_label]:
            unified_data[assigned_label][epic_key] = {
                'epic_name': epic_info[1],
                'worklogs': {},
                'planned_tasks': []
            }
        
        # Add the planned task to the epic
        unified_data[assigned_label][epic_key]['planned_tasks'].append(task)

    # Generate the HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        ul {{ list-style-type: none; padding-left: 0; }}
        .task {{ margin-left: 20px; list-style-type: circle; }}
        .subtask {{ margin-left: 40px; list-style-type: square; }}
        .worklog {{ margin-left: 20px; font-style: italic; color: #666; }}
        .author {{ color: #888; font-size: 0.9em; }}
        .label-cell {{
            width: 100px;
            text-align: center;
            font-weight: bold;
            color: white;
        }}
        .label-B2C {{ background-color: #3498db; }}
        .label-B2B {{ background-color: #e74c3c; }}
        .label-Licensing {{ background-color: #2ecc71; }}
        .label-RI {{ background-color: #f1c40f; }}
        .label-NoLabel {{ background-color: #95a5a6; }}
        .charts-container {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }}
        .chart {{
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
        }}
        .planned-tasks-header {{
            margin-top: 10px;
            font-weight: bold;
            color: #666;
            border-bottom: 1px solid #ddd;
        }}
        .task.planned {{
            background-color: #f8f9fa;
        }}
    </style>
    </head>
    <body>
    <div class="charts-container">
        <div class="chart">
            <img src="data:image/png;base64,{chart1}" alt="Distribution excluding B2C/B2B/RI">
        </div>
        <div class="chart">
            <img src="data:image/png;base64,{chart2}" alt="Distribution B2C/B2B/RI">
        </div>
    </div>
    <table>
    """

    for label, epics in unified_data.items():
        if epics:  # If the label contains epics
            label_class = f"label-{label.replace(' ', '')}"
            
            # Count the total number of rows for this label
            total_rows = len(epics)
            
            first_epic = True
            for epic_key, epic_data in epics.items():
                if not first_epic:
                    html += "<tr>"
                else:
                    html += f"""
                    <tr>
                        <td class="label-cell {label_class}" rowspan="{total_rows}">{label}</td>
                    """
                
                epic_name = epic_data['epic_name']
                details = "<ul>"
                
                # Add the header "Progress:"
                details += '<li class="planned-tasks-header">Progress:</li>'
                
                # Add the worklogs
                for issue_key, issue_data in epic_data['worklogs'].items():
                    issue_summary = issue_data['Summary']
                    details += f'<li class="task"><a href="https://devialet.atlassian.net/browse/{issue_key}">{issue_key}</a>: {issue_summary}'
                    
                    # Group worklogs by whether they have comments
                    worklogs_with_comments = []
                    authors_without_comments = set()
                    
                    for worklog in issue_data['Worklogs']:
                        if worklog['Worklog Comment']:
                            worklogs_with_comments.append(worklog)
                        else:
                            author = worklog['Worklog Author']
                            initials = ''.join(word[0].upper() for word in author.split())
                            authors_without_comments.add(initials)
                    
                    if authors_without_comments:
                        details += f' <span class="author">[{", ".join(sorted(authors_without_comments))}]</span>'
                    
                    for worklog in worklogs_with_comments:
                        author = worklog['Worklog Author']
                        initials = ''.join(word[0].upper() for word in author.split())
                        details += f'<div class="worklog">{worklog["Worklog Comment"]} [{initials}]</div>'
                    
                    details += '</li>'
                
                # Add the planned tasks
                if epic_data['planned_tasks']:
                    # Change "Planned Tasks:" to "Plan:"
                    details += '<li class="planned-tasks-header">Plan:</li>'
                    for task in epic_data['planned_tasks']:
                        assignee_initials = ''.join(word[0].upper() for word in task['Assignee'].split()) if task['Assignee'] != 'Unassigned' else 'UA'
                        details += f'''
                            <li class="task planned">
                                <a href="https://devialet.atlassian.net/browse/{task['Key']}">{task['Key']}</a>: 
                                {task['Summary']} 
                                <span class="author">[{assignee_initials}]</span>
                                <span style="color: #666;">({task['Status']})</span>
                            </li>
                        '''
                
                details += "</ul>"
                
                html += f"""
                    <td>{epic_name}</td>
                    <td>{details}</td>
                </tr>
                """
                first_epic = False

    html += """
    </table>
    </body>
    </html>
    """

    return html

def get_planned_tasks(jira, project):
    """
    Retrieve all tasks, sub-tasks, and stories in To Do or In Progress status.
    
    :param jira: Connected JIRA instance
    :param project: JIRA project key
    :return: DataFrame with planned tasks
    """
    # JQL to retrieve planned tasks
    jql_planned = f'''project = {project} 
        AND status in ("To Do", "In Progress") 
        AND issuetype in (Story, Sub-task, Task)
        ORDER BY status ASC, created DESC'''

    try:
        # Fields to retrieve
        fields = 'key,summary,status,issuetype,priority,assignee,customfield_10008'  # customfield_10008 is the Epic Link
        
        # Retrieve issues
        planned_issues = jira.search_issues(jql_planned, fields=fields, maxResults=False)
        logging.info(f"Retrieved {len(planned_issues)} planned issues")

        # Prepare data for the DataFrame
        planned_data = []
        for issue in planned_issues:
            # Retrieve the epic
            epic_key = getattr(issue.fields, 'customfield_10008', None)
            epic_name = ''
            if epic_key:
                try:
                    epic = jira.issue(epic_key)
                    epic_name = epic.fields.summary
                except:
                    epic_name = 'Unknown Epic'

            # Retrieve the assignee
            assignee = getattr(issue.fields.assignee, 'displayName', 'Unassigned') if issue.fields.assignee else 'Unassigned'

            planned_data.append({
                'Key': issue.key,
                'Type': issue.fields.issuetype.name,
                'Status': issue.fields.status.name,
                'Priority': issue.fields.priority.name,
                'Summary': issue.fields.summary,
                'Assignee': assignee,
                'Epic': f"{epic_key} - {epic_name}" if epic_key else 'No Epic'
            })

        # Create the DataFrame
        df = pd.DataFrame(planned_data)
        
        # Display the table in the console
        print("\n=== PLANNED TASKS ===")
        print(df.to_string(index=False))
        
        return df

    except Exception as e:
        logging.error(f"Error retrieving planned tasks: {e}")
        return pd.DataFrame()

def send_html_email(html_content, start_date, end_date):
    """
    Send the HTML report via email.
    """
    # Retrieve email information from environment variables
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    recipient_emails = os.getenv('RECIPIENT_EMAILS').split(',')  # List of emails separated by commas

    # Create the message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f'ASD Activity Report ({start_date} to {end_date})'
    msg['From'] = sender_email
    msg['To'] = ', '.join(recipient_emails)

    # Attach the HTML content
    msg.attach(MIMEText(html_content, 'html'))

    try:
        # Connect to the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logging.info("Email sent successfully")
        return True
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        return False

def main():
    # Load environment variables
    jira_server = os.getenv('JIRA_SERVER')
    jira_email = os.getenv('JIRA_EMAIL')
    jira_token = os.getenv('JIRA_TOKEN')
    epic_field_id = os.getenv('JIRA_EPIC_FIELD_ID', 'customfield_10008')
    email_recipient = os.getenv('EMAIL_RECIPIENT')

    # Validate environment variables
    required_env_vars = ['JIRA_SERVER', 'JIRA_EMAIL', 'JIRA_TOKEN']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    try:
        logging.info("Connecting to JIRA...")
        jira = get_jira_connection(jira_server, jira_email, jira_token)
        logging.info("Successfully connected to JIRA.")

        project_key = 'ASD'  # Replace with your project key or make it configurable
        # Get dates from last Monday to this Friday
        today = datetime.now()
        start_date = (today - timedelta(days=today.weekday() + 7)).strftime('%Y-%m-%d')  # Last Monday
        end_date = (today + timedelta(days=4 - today.weekday())).strftime('%Y-%m-%d')    # This Friday

        # Retrieve planned tasks
        planned_tasks_df = get_planned_tasks(jira, project_key)

        logging.info(f"Extracting worklogs for project {project_key} from {start_date} to {end_date}.")
        worklogs_df = extract_worklogs_in_period(jira, project_key, start_date, end_date, epic_field_id)
        total_time_per_person_and_label = calculate_time_per_person_and_label(worklogs_df)
        total_time_per_person_and_epic = calculate_time_per_person_and_epic(worklogs_df)
        filtered_df_exclude, filtered_df_include = filter_time_by_label(
            total_time_per_person_and_label, 
            ['B2C', 'B2B', 'RI', 'Licensing']
        )
        #create_graph_from_filtered_df(filtered_df_exclude)
        #create_graph_from_filtered_df(filtered_df_include)

        # Display the table
        print(worklogs_df)

        # Optionally: save to CSV
        csv_file = 'jira_worklogs.csv'
        worklogs_df.to_csv(csv_file, index=False)
        logging.info(f"Worklogs saved to {csv_file}.")

        if not worklogs_df.empty:
            # Convert DataFrame to JSON
            worklogs_json = convert_worklogs_to_json(worklogs_df)
            csv_output = create_csv_from_json(worklogs_json)
            
            # Calculate data for charts
            total_time_per_person_and_label = calculate_time_per_person_and_label(worklogs_df)
            filtered_df_exclude, filtered_df_include = filter_time_by_label(
                total_time_per_person_and_label, 
                ['B2C', 'B2B', 'RI', 'Licensing']
            )
            
            # Create the HTML table with charts and planned tasks
            html_table = create_html_table_from_json(
                worklogs_json,
                filtered_df_exclude,
                filtered_df_include,
                planned_tasks_df
            )

            print(csv_output)
            # Save HTML to file
            html_file = 'worklogs_table.html'
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_table)
            logging.info(f"HTML table saved to {html_file}")
            # Print rendered HTML table to console
            from bs4 import BeautifulSoup
            
            # Parse and prettify the HTML for better console display
            soup = BeautifulSoup(html_table, 'html.parser')
            pretty_html = soup.prettify()
            
            print("\nRendered HTML Output:")
            print(pretty_html)

            # Display the JSON
            print(worklogs_json)

            # Optionally: save to a JSON file
            json_file = 'jira_worklogs.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(worklogs_json)
            logging.info(f"Worklogs JSON saved to {json_file}.")

            # Send the email with the HTML report
            email_sent = send_html_email(html_table, start_date, end_date)
            
            if not email_sent:
                logging.error("Failed to send the email")
                sys.exit(1)
        else:
            logging.info("No worklogs found for the specified period.")

    except Exception as e:
        logging.exception(f"Error connecting to JIRA or processing worklogs: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

